"""Training and evaluation orchestration."""

from __future__ import annotations

import math
import time
from typing import Any, Iterable, List, MutableMapping, Optional, Union

import mlx.core as mx

from .backend.amp import AMPConfig, GradScaler, cast_to_dtype
from .network import Network
from .pipeline import Pipeline
from .summary import Summary


class Estimator:
    """Run a :class:`~fastmlx.pipeline.Pipeline` using a :class:`~fastmlx.network.Network`.

    Args:
        pipeline: The data pipeline for loading and preprocessing data.
        network: The network containing ops to execute.
        epochs: Number of training epochs.
        traces: Optional list of trace callbacks.
        log_interval: How often to log training progress (in steps).
        verbose: Verbosity level. Options:
                 - 0: Silent (no output)
                 - 1: Progress bar only (epoch-level)
                 - 2: Normal output (default, logs at log_interval)
                 - 3: Verbose (logs every batch)
        mixed_precision: Enable mixed precision training. Can be:
                        - False/None: Disabled (default)
                        - True: Enable with float16
                        - "float16" or "fp16": Enable with float16
                        - "bfloat16" or "bf16": Enable with bfloat16
                        - AMPConfig: Custom AMP configuration

    Example:
        >>> estimator = Estimator(
        ...     pipeline=pipeline,
        ...     network=network,
        ...     epochs=10,
        ...     traces=[Accuracy(true_key="y", pred_key="y_pred")]
        ... )
        >>> estimator.fit()

        >>> # Silent training
        >>> estimator = Estimator(..., verbose=0)

        >>> # Verbose training
        >>> estimator = Estimator(..., verbose=3)

        >>> # Mixed precision training
        >>> estimator = Estimator(..., mixed_precision=True)
        >>> estimator = Estimator(..., mixed_precision="bfloat16")

        >>> # Named experiment for tracking
        >>> estimator = Estimator(..., experiment_name="cifar10_resnet")
        >>> summary = estimator.fit()
        >>> summary.save("results/")
    """

    def __init__(
        self,
        pipeline: Pipeline,
        network: Network,
        epochs: int,
        traces: Optional[Iterable[object]] = None,
        log_interval: int = 100,
        verbose: int = 2,
        mixed_precision: Union[bool, str, AMPConfig, None] = None,
        experiment_name: str = "experiment",
    ) -> None:
        self.pipeline: Pipeline = pipeline
        self.network: Network = network
        self.epochs: int = epochs
        self.traces: List[object] = list(traces or [])
        self.log_interval = log_interval
        self.verbose = verbose
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.experiment_name: str = experiment_name

        # Create summary for tracking metrics
        self.summary: Summary = Summary(name=experiment_name)

        # Setup mixed precision
        self.amp_config: Optional[AMPConfig] = self._setup_amp(mixed_precision)
        self.grad_scaler: Optional[GradScaler] = None
        if self.amp_config and self.amp_config.enabled:
            self.grad_scaler = GradScaler() if self.amp_config.grad_scale else None
            self._apply_amp_to_network()

    def _setup_amp(
        self,
        mixed_precision: Union[bool, str, AMPConfig, None]
    ) -> Optional[AMPConfig]:
        """Parse mixed_precision argument into AMPConfig."""
        if mixed_precision is None or mixed_precision is False:
            return None

        if isinstance(mixed_precision, AMPConfig):
            return mixed_precision

        if mixed_precision is True or mixed_precision in ("float16", "fp16"):
            return AMPConfig(enabled=True, dtype=mx.float16)

        if mixed_precision in ("bfloat16", "bf16"):
            return AMPConfig(enabled=True, dtype=mx.bfloat16)

        raise ValueError(
            f"Invalid mixed_precision value: {mixed_precision}. "
            f"Expected True, False, 'float16', 'bfloat16', or AMPConfig."
        )

    def _apply_amp_to_network(self) -> None:
        """Apply AMP settings to network ops (cast models to target dtype)."""
        if not self.amp_config or not self.amp_config.enabled:
            return

        from .op.model_op import ModelOp
        from .op.update_op import UpdateOp

        dtype = self.amp_config.dtype
        casted_models = set()

        for op in self.network.ops:
            if isinstance(op, (ModelOp, UpdateOp)) and hasattr(op, "model"):
                model_id = id(op.model)
                if model_id not in casted_models:
                    cast_to_dtype(op.model, dtype)
                    casted_models.add(model_id)
                    self._log(
                        f"AMP: Cast {op.model.__class__.__name__} to {dtype}",
                        level=2
                    )

            # Pass grad_scaler to UpdateOp if using gradient scaling
            if isinstance(op, UpdateOp) and self.grad_scaler is not None:
                op.grad_scaler = self.grad_scaler

    def _log(self, message: str, level: int = 2) -> None:
        """Log message if verbosity level allows."""
        if self.verbose >= level:
            print(message)

    def _get_learning_rate(self) -> Optional[float]:
        """Safely get the current learning rate from the network's model ops."""
        for op in self.network.ops:
            if hasattr(op, "model") and hasattr(op.model, "optimizer"):
                lr = getattr(op.model.optimizer, "learning_rate", None)
                if lr is not None:
                    return float(lr) if isinstance(lr, (int, float)) else lr
        return None

    def _get_loss_value(self, batch: MutableMapping[str, Any]) -> Optional[float]:
        """Extract loss value from batch using network's loss keys."""
        loss_keys = self.network.get_loss_keys()
        for key in loss_keys:
            if key in batch:
                loss_val = batch[key]
                if isinstance(loss_val, mx.array):
                    return float(loss_val.item())
                elif isinstance(loss_val, (int, float)):
                    return float(loss_val)
        return None

    def _get_loss_key_name(self) -> str:
        """Get the primary loss key name for logging."""
        loss_keys = list(self.network.get_loss_keys())
        return loss_keys[0] if loss_keys else "loss"

    def _should_stop(self, state: MutableMapping[str, Any]) -> bool:
        """Check if training should stop (e.g., from EarlyStopping)."""
        return state.get("should_stop", False)

    def _trace_should_run(self, trace: object, mode: Optional[str]) -> bool:
        """Check if a trace should run in the given mode.

        Args:
            trace: The trace object.
            mode: Current execution mode.

        Returns:
            True if trace should run, False otherwise.
        """
        if hasattr(trace, "should_run"):
            return trace.should_run(mode)
        return True  # Default to running if no should_run method

    def _get_dataset_length(self, dataset: Any) -> Optional[int]:
        """Safely get dataset length."""
        try:
            return len(dataset)
        except (TypeError, AttributeError):
            return None

    def _cast_batch_for_amp(
        self,
        batch: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Cast batch data to AMP dtype if enabled.

        Only casts float arrays; keeps integer arrays (like labels) unchanged.
        """
        if not self.amp_config or not self.amp_config.enabled:
            return batch

        dtype = self.amp_config.dtype
        for key, value in batch.items():
            if isinstance(value, mx.array):
                # Only cast float types, leave integers alone (e.g., labels)
                if value.dtype in (mx.float32, mx.float64):
                    batch[key] = value.astype(dtype)

        return batch

    def _warmup(self) -> None:
        """Validate the configuration by running a single batch.

        This catches configuration errors (missing keys, shape mismatches)
        before the full training loop starts.

        Raises:
            Various exceptions if configuration is invalid.
        """
        self._log("FastMLX: Running warmup validation...", level=2)

        # Validate network ops
        warnings = self.network.validate_ops()
        for warning in warnings:
            self._log(f"FastMLX-Warning: {warning}", level=1)

        # Get a single batch from training data
        loader = self.pipeline.get_loader("train")
        try:
            batch = next(iter(loader))
        except StopIteration:
            raise ValueError("Pipeline returned no data. Check your train_data.")

        # Cast for AMP if enabled
        batch = self._cast_batch_for_amp(batch)

        # Run through network in train mode
        state: MutableMapping[str, Any] = {"mode": "train", "epoch": 0}

        try:
            self.network.run(batch, state)
        except Exception as e:
            raise RuntimeError(
                f"Warmup validation failed during training forward pass: {e}\n"
                f"Available batch keys: {list(batch.keys())}"
            ) from e

        # Also validate eval mode if eval_data exists
        if self.pipeline.eval_data is not None:
            eval_loader = self.pipeline.get_loader("eval")
            try:
                eval_batch = next(iter(eval_loader))
            except StopIteration:
                self._log("FastMLX-Warning: eval_data returned no batches", level=1)
                return

            eval_batch = self._cast_batch_for_amp(eval_batch)
            eval_state: MutableMapping[str, Any] = {"mode": "eval", "epoch": 0}

            try:
                self.network.run(eval_batch, eval_state)
            except Exception as e:
                raise RuntimeError(
                    f"Warmup validation failed during eval forward pass: {e}\n"
                    f"Available batch keys: {list(eval_batch.keys())}"
                ) from e

        self._log("FastMLX: Warmup validation passed.", level=2)

    def fit(self, warmup: bool = True) -> Summary:
        """Train the network with periodic logging.

        Args:
            warmup: If True (default), validate configuration before training
                by running a single batch through the network.

        Returns:
            Summary object containing all metrics history from training.
            Use summary.get_values("loss", mode="train") to access specific metrics,
            or summary.save("results/") to persist the results.

        Note:
            Training can be stopped early by traces that set
            ``state['should_stop'] = True`` (e.g., EarlyStopping).
        """
        # Run warmup validation
        if warmup:
            self._warmup()

        state: MutableMapping[str, object] = {"should_stop": False}
        step = self.global_step
        start_time = time.time()

        self._log(
            f"FastMLX-Start: step: 1; logging_interval: {self.log_interval}; num_device: 1;",
            level=1
        )

        # Call on_start for all traces
        for t in self.traces:
            if hasattr(t, "on_start"):
                t.on_start(state)

        for epoch in range(self.epochs):
            # Check for early stopping before starting new epoch
            if self._should_stop(state):
                self._log(f"FastMLX: Training stopped early at epoch {epoch}", level=1)
                break

            epoch_start = time.time()
            self.current_epoch = epoch + 1
            state.update({"mode": "train", "epoch": epoch, "metrics": {}})

            # Epoch begin callbacks
            for t in self.traces:
                if hasattr(t, "on_epoch_begin") and self._trace_should_run(t, "train"):
                    t.on_epoch_begin(state)

            # Training loop
            for batch in self.pipeline.get_loader("train"):
                step += 1
                self.global_step += 1
                batch_start = time.time()

                # Cast batch data for mixed precision
                batch = self._cast_batch_for_amp(batch)
                state["batch"] = batch

                # Batch begin callbacks
                for t in self.traces:
                    if hasattr(t, "on_batch_begin") and self._trace_should_run(t, "train"):
                        t.on_batch_begin(batch, state)

                # Run network forward/backward
                try:
                    self.network.run(batch, state)
                except Exception as e:
                    self._log(f"FastMLX-Error: step {step}; error: {e}", level=0)
                    raise

                # Batch end callbacks
                for t in self.traces:
                    if hasattr(t, "on_batch_end") and self._trace_should_run(t, "train"):
                        t.on_batch_end(batch, state)

                # Check for early stopping after each batch
                if self._should_stop(state):
                    break

                # Verbose logging (every batch)
                if self.verbose >= 3:
                    lr = self._get_learning_rate()
                    loss_val = self._get_loss_value(batch)
                    loss_key = self._get_loss_key_name()
                    steps_per_sec = 1.0 / max(time.time() - batch_start, 1e-8)
                    self._log(
                        f"FastMLX-Train: step: {step}; {loss_key}: {loss_val}; "
                        f"model_lr: {lr}; steps/sec: {steps_per_sec:.2f};",
                        level=3
                    )
                # Normal periodic logging
                elif step % self.log_interval == 0:
                    lr = self._get_learning_rate()
                    loss_val = self._get_loss_value(batch)
                    loss_key = self._get_loss_key_name()
                    steps_per_sec = 1.0 / max(time.time() - batch_start, 1e-8)
                    self._log(
                        f"FastMLX-Train: step: {step}; {loss_key}: {loss_val}; "
                        f"model_lr: {lr}; steps/sec: {steps_per_sec:.2f};",
                        level=2
                    )

            # Epoch end callbacks
            for t in self.traces:
                if hasattr(t, "on_epoch_end") and self._trace_should_run(t, "train"):
                    t.on_epoch_end(state)

            # Record training metrics to summary
            metrics = state.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.summary.add(
                        metric_name, metric_value,
                        epoch=epoch, step=step, mode="train"
                    )

            epoch_time = time.time() - epoch_start
            self._log(
                f"FastMLX-Train: step: {step}; epoch: {epoch+1}; epoch_time: {epoch_time:.2f} sec;",
                level=1
            )

            # Check for early stopping after epoch
            if self._should_stop(state):
                self._log(f"FastMLX: Training stopped early after epoch {epoch + 1}", level=1)
                break

            # Evaluation phase
            if self.pipeline.eval_data is not None:
                self._run_evaluation(step, epoch, state)

        # Finish callbacks
        for t in self.traces:
            if hasattr(t, "on_finish"):
                t.on_finish(state)

        total_time = time.time() - start_time
        lr = self._get_learning_rate()
        self._log(
            f"FastMLX-Finish: step: {step}; model_lr: {lr}; total_time: {total_time:.2f} sec;",
            level=1
        )

        # Store final state in summary metadata
        self.summary.metadata["total_time"] = total_time
        self.summary.metadata["final_step"] = step
        self.summary.metadata["final_epoch"] = self.current_epoch
        if lr is not None:
            self.summary.metadata["final_lr"] = lr

        return self.summary

    def _run_evaluation(
        self,
        step: int,
        epoch: int,
        train_state: MutableMapping[str, object]
    ) -> None:
        """Run evaluation phase after training epoch."""
        eval_state: MutableMapping[str, object] = {
            "mode": "eval",
            "epoch": epoch,
            "metrics": {},
            "should_stop": train_state.get("should_stop", False),
        }

        # Epoch begin for eval
        for t in self.traces:
            if hasattr(t, "on_epoch_begin") and self._trace_should_run(t, "eval"):
                t.on_epoch_begin(eval_state)

        eval_dataset = self.pipeline.eval_data
        dataset_len = self._get_dataset_length(eval_dataset)
        num_batches = (
            math.ceil(dataset_len / self.pipeline.batch_size)
            if dataset_len is not None
            else None
        )

        eval_step = 0
        total_loss = 0.0
        loss_count = 0

        for batch in self.pipeline.get_loader("eval"):
            eval_step += 1
            batch_start = time.time()

            # Cast batch data for mixed precision
            batch = self._cast_batch_for_amp(batch)
            eval_state["batch"] = batch

            # Batch begin callbacks
            for t in self.traces:
                if hasattr(t, "on_batch_begin") and self._trace_should_run(t, "eval"):
                    t.on_batch_begin(batch, eval_state)

            try:
                self.network.run(batch, eval_state)
            except Exception as e:
                self._log(f"FastMLX-Eval-Error: step {eval_step}; error: {e}", level=0)
                raise

            # Batch end callbacks
            for t in self.traces:
                if hasattr(t, "on_batch_end") and self._trace_should_run(t, "eval"):
                    t.on_batch_end(batch, eval_state)

            # Track loss
            loss_val = self._get_loss_value(batch)
            if loss_val is not None:
                total_loss += loss_val
                loss_count += 1

            # Progress logging
            should_log = (
                eval_step == 1 or
                eval_step % self.log_interval == 0 or
                (num_batches is not None and eval_step == num_batches)
            )
            if should_log:
                steps_per_sec = 1.0 / max(time.time() - batch_start, 1e-8)
                progress = f"{eval_step}/{num_batches}" if num_batches else f"{eval_step}"
                self._log(f"Eval Progress: {progress}; steps/sec: {steps_per_sec:.2f};", level=2)

        # Epoch end for eval
        for t in self.traces:
            if hasattr(t, "on_epoch_end") and self._trace_should_run(t, "eval"):
                t.on_epoch_end(eval_state)

        # Record average loss
        loss_key = self._get_loss_key_name()
        if loss_count:
            eval_state["metrics"][loss_key] = total_loss / loss_count

        # Record eval metrics to summary
        eval_metrics = eval_state.get("metrics", {})
        for metric_name, metric_value in eval_metrics.items():
            if isinstance(metric_value, (int, float)):
                self.summary.add(
                    metric_name, metric_value,
                    epoch=epoch, step=step, mode="eval"
                )

        acc = eval_state["metrics"].get("accuracy")
        loss_metric = eval_state["metrics"].get(loss_key)
        self._log(
            f"FastMLX-Eval: step: {step}; epoch: {epoch+1}; accuracy: {acc}; {loss_key}: {loss_metric};",
            level=1
        )

        # Propagate should_stop from eval to train state
        if eval_state.get("should_stop"):
            train_state["should_stop"] = True

    def test(self) -> MutableMapping[str, object]:
        """Evaluate the network on the eval dataset.

        Returns:
            Evaluation state dictionary containing metrics.
        """
        state: MutableMapping[str, object] = {"mode": "eval", "metrics": {}}

        for t in self.traces:
            if hasattr(t, "on_epoch_begin") and self._trace_should_run(t, "eval"):
                t.on_epoch_begin(state)

        total_loss = 0.0
        loss_count = 0
        step = 0

        for batch in self.pipeline.get_loader("eval"):
            step += 1

            # Cast batch data for mixed precision
            batch = self._cast_batch_for_amp(batch)
            state["batch"] = batch

            # Batch begin callbacks
            for t in self.traces:
                if hasattr(t, "on_batch_begin") and self._trace_should_run(t, "eval"):
                    t.on_batch_begin(batch, state)

            try:
                self.network.run(batch, state)
            except Exception as e:
                self._log(f"FastMLX-Test-Error: step {step}; error: {e}", level=0)
                raise

            # Batch end callbacks
            for t in self.traces:
                if hasattr(t, "on_batch_end") and self._trace_should_run(t, "eval"):
                    t.on_batch_end(batch, state)

            loss_val = self._get_loss_value(batch)
            if loss_val is not None:
                total_loss += loss_val
                loss_count += 1

        for t in self.traces:
            if hasattr(t, "on_epoch_end") and self._trace_should_run(t, "eval"):
                t.on_epoch_end(state)

        loss_key = self._get_loss_key_name()
        if loss_count:
            state["metrics"][loss_key] = total_loss / loss_count

        acc = state["metrics"].get("accuracy")
        loss_metric = state["metrics"].get(loss_key)
        self._log(
            f"FastMLX-Test: step: {self.global_step}; epoch: {self.current_epoch}; "
            f"accuracy: {acc}; {loss_key}: {loss_metric};",
            level=1
        )
        return state
