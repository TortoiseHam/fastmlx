"""Training and evaluation orchestration."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, MutableMapping, Optional
import math
import time

import mlx.core as mx

from .pipeline import Pipeline
from .network import Network


class Estimator:
    """Run a :class:`~fastmlx.pipeline.Pipeline` using a :class:`~fastmlx.network.Network`.

    Args:
        pipeline: The data pipeline for loading and preprocessing data.
        network: The network containing ops to execute.
        epochs: Number of training epochs.
        traces: Optional list of trace callbacks.
        log_interval: How often to log training progress (in steps).

    Example:
        >>> estimator = Estimator(
        ...     pipeline=pipeline,
        ...     network=network,
        ...     epochs=10,
        ...     traces=[Accuracy(true_key="y", pred_key="y_pred")]
        ... )
        >>> estimator.fit()
    """

    def __init__(
        self,
        pipeline: Pipeline,
        network: Network,
        epochs: int,
        traces: Optional[Iterable[object]] = None,
        log_interval: int = 100
    ) -> None:
        self.pipeline: Pipeline = pipeline
        self.network: Network = network
        self.epochs: int = epochs
        self.traces: List[object] = list(traces or [])
        self.log_interval = log_interval
        self.global_step: int = 0
        self.current_epoch: int = 0

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

    def _get_dataset_length(self, dataset: Any) -> Optional[int]:
        """Safely get dataset length."""
        try:
            return len(dataset)
        except (TypeError, AttributeError):
            return None

    def fit(self) -> MutableMapping[str, object]:
        """Train the network with periodic logging.

        Returns:
            Final training state dictionary containing metrics.

        Note:
            Training can be stopped early by traces that set
            ``state['should_stop'] = True`` (e.g., EarlyStopping).
        """
        state: MutableMapping[str, object] = {"should_stop": False}
        step = self.global_step
        start_time = time.time()

        print(
            f"FastMLX-Start: step: 1; logging_interval: {self.log_interval}; num_device: 1;"
        )

        # Call on_start for all traces
        for t in self.traces:
            if hasattr(t, "on_start"):
                t.on_start(state)

        for epoch in range(self.epochs):
            # Check for early stopping before starting new epoch
            if self._should_stop(state):
                print(f"FastMLX: Training stopped early at epoch {epoch}")
                break

            epoch_start = time.time()
            self.current_epoch = epoch + 1
            state.update({"mode": "train", "epoch": epoch, "metrics": {}})

            # Epoch begin callbacks
            for t in self.traces:
                if hasattr(t, "on_epoch_begin"):
                    t.on_epoch_begin(state)

            # Training loop
            for batch in self.pipeline.get_loader("train"):
                step += 1
                self.global_step += 1
                batch_start = time.time()
                state["batch"] = batch

                # Run network forward/backward
                try:
                    self.network.run(batch, state)
                except Exception as e:
                    print(f"FastMLX-Error: step {step}; error: {e}")
                    raise

                # Batch end callbacks
                for t in self.traces:
                    if hasattr(t, "on_batch_end"):
                        t.on_batch_end(batch, state)

                # Check for early stopping after each batch
                if self._should_stop(state):
                    break

                # Periodic logging
                if step % self.log_interval == 0:
                    lr = self._get_learning_rate()
                    loss_val = self._get_loss_value(batch)
                    loss_key = self._get_loss_key_name()
                    steps_per_sec = 1.0 / max(time.time() - batch_start, 1e-8)
                    print(
                        f"FastMLX-Train: step: {step}; {loss_key}: {loss_val}; "
                        f"model_lr: {lr}; steps/sec: {steps_per_sec:.2f};"
                    )

            # Epoch end callbacks
            for t in self.traces:
                if hasattr(t, "on_epoch_end"):
                    t.on_epoch_end(state)

            epoch_time = time.time() - epoch_start
            print(f"FastMLX-Train: step: {step}; epoch: {epoch+1}; epoch_time: {epoch_time:.2f} sec;")

            # Check for early stopping after epoch
            if self._should_stop(state):
                print(f"FastMLX: Training stopped early after epoch {epoch + 1}")
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
        print(
            f"FastMLX-Finish: step: {step}; model_lr: {lr}; total_time: {total_time:.2f} sec;"
        )
        return state

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
            if hasattr(t, "on_epoch_begin"):
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
            eval_state["batch"] = batch

            try:
                self.network.run(batch, eval_state)
            except Exception as e:
                print(f"FastMLX-Eval-Error: step {eval_step}; error: {e}")
                raise

            # Batch end callbacks
            for t in self.traces:
                if hasattr(t, "on_batch_end"):
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
                print(f"Eval Progress: {progress}; steps/sec: {steps_per_sec:.2f};")

        # Epoch end for eval
        for t in self.traces:
            if hasattr(t, "on_epoch_end"):
                t.on_epoch_end(eval_state)

        # Record average loss
        loss_key = self._get_loss_key_name()
        if loss_count:
            eval_state["metrics"][loss_key] = total_loss / loss_count

        acc = eval_state["metrics"].get("accuracy")
        loss_metric = eval_state["metrics"].get(loss_key)
        print(
            f"FastMLX-Eval: step: {step}; epoch: {epoch+1}; accuracy: {acc}; {loss_key}: {loss_metric};"
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
            if hasattr(t, "on_epoch_begin"):
                t.on_epoch_begin(state)

        total_loss = 0.0
        loss_count = 0
        step = 0

        for batch in self.pipeline.get_loader("eval"):
            step += 1
            state["batch"] = batch

            try:
                self.network.run(batch, state)
            except Exception as e:
                print(f"FastMLX-Test-Error: step {step}; error: {e}")
                raise

            for t in self.traces:
                if hasattr(t, "on_batch_end"):
                    t.on_batch_end(batch, state)

            loss_val = self._get_loss_value(batch)
            if loss_val is not None:
                total_loss += loss_val
                loss_count += 1

        for t in self.traces:
            if hasattr(t, "on_epoch_end"):
                t.on_epoch_end(state)

        loss_key = self._get_loss_key_name()
        if loss_count:
            state["metrics"][loss_key] = total_loss / loss_count

        acc = state["metrics"].get("accuracy")
        loss_metric = state["metrics"].get(loss_key)
        print(
            f"FastMLX-Test: step: {self.global_step}; epoch: {self.current_epoch}; "
            f"accuracy: {acc}; {loss_key}: {loss_metric};"
        )
        return state
