"""Adaptive training traces for learning rate scheduling and early stopping."""

from __future__ import annotations

import math
from typing import Callable, MutableMapping, Optional

from .base import Trace


class LRScheduler(Trace):
    """Adjust learning rate based on a scheduling function.

    Args:
        model: The model whose optimizer's learning rate will be adjusted.
        lr_fn: A function that takes the current step and returns the learning rate.
    """

    def __init__(self, model, lr_fn: Callable[[int], float]) -> None:
        self.model = model
        self.lr_fn = lr_fn
        self.step: int = 0

    def on_start(self, state: MutableMapping[str, object]) -> None:
        self.step = 0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        self.step += 1
        lr = self.lr_fn(self.step)
        self.model.optimizer.learning_rate = lr


class EarlyStopping(Trace):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric name to monitor (from state['metrics']).
        min_delta: Minimum change to qualify as an improvement.
        patience: Number of epochs with no improvement after which training will be stopped.
        mode: One of 'min' or 'max'. In 'min' mode, training stops when the
              metric stops decreasing; in 'max' mode it stops when the metric
              stops increasing.
        baseline: Baseline value for the monitored metric. Training will stop if
                  the metric doesn't improve beyond this baseline.
        restore_best_weights: Whether to restore model weights from the epoch with
                              the best value of the monitored metric.
        model: The model to save/restore weights for. Required if restore_best_weights=True.

    Example:
        >>> # Basic early stopping
        >>> EarlyStopping(monitor="accuracy", patience=5, mode="max")

        >>> # With weight restoration
        >>> EarlyStopping(
        ...     monitor="eval_loss",
        ...     patience=10,
        ...     mode="min",
        ...     restore_best_weights=True,
        ...     model=model
        ... )
    """

    def __init__(
        self,
        monitor: str = "loss",
        min_delta: float = 0.0,
        patience: int = 5,
        mode: str = "min",
        baseline: Optional[float] = None,
        restore_best_weights: bool = False,
        model: Optional[object] = None,
    ) -> None:
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.model = model

        # Validate that model is provided if restore_best_weights is True
        if restore_best_weights and model is None:
            raise ValueError(
                "EarlyStopping: model must be provided when restore_best_weights=True"
            )

        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.best: Optional[float] = None
        self.best_weights: Optional[dict] = None

        if mode == "min":
            self.monitor_op = lambda a, b: a < b - min_delta
        else:
            self.monitor_op = lambda a, b: a > b + min_delta

    def _save_weights(self) -> None:
        """Save current model weights."""
        if self.model is not None and hasattr(self.model, 'parameters'):
            # Deep copy the parameters
            import copy
            self.best_weights = copy.deepcopy(dict(self.model.parameters()))

    def _restore_weights(self) -> None:
        """Restore saved model weights."""
        if self.model is not None and self.best_weights is not None:
            if hasattr(self.model, 'update'):
                self.model.update(self.best_weights)

    def on_start(self, state: MutableMapping[str, object]) -> None:
        self.wait = 0
        self.stopped_epoch = 0
        self.best = self.baseline
        self.best_weights = None

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        current = state['metrics'].get(self.monitor)
        if current is None:
            return

        epoch = state.get('epoch', 0)

        if self.best is None:
            self.best = current
            if self.restore_best_weights:
                self._save_weights()
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self._save_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                state['should_stop'] = True
                print(f"FastMLX-EarlyStopping: Stopping training at epoch {epoch + 1}")
                if self.restore_best_weights and self.best_weights is not None:
                    print("FastMLX-EarlyStopping: Restoring best weights")
                    self._restore_weights()


class ReduceLROnPlateau(Trace):
    """Reduce learning rate when a metric has stopped improving.

    Args:
        model: The model whose optimizer's learning rate will be adjusted.
        monitor: Metric name to monitor (from state['metrics']).
        factor: Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience: Number of epochs with no improvement after which LR will be reduced.
        mode: One of 'min' or 'max'. In 'min' mode, LR is reduced when the
              metric stops decreasing; in 'max' mode when it stops increasing.
        min_delta: Threshold for measuring the new optimum.
        cooldown: Number of epochs to wait before resuming normal operation after
                  LR has been reduced.
        min_lr: Lower bound on the learning rate.
    """

    def __init__(
        self,
        model,
        monitor: str = "loss",
        factor: float = 0.1,
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0.0
    ) -> None:
        self.model = model
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr

        self.wait: int = 0
        self.cooldown_counter: int = 0
        self.best: Optional[float] = None

        if mode == "min":
            self.monitor_op = lambda a, b: a < b - min_delta
        else:
            self.monitor_op = lambda a, b: a > b + min_delta

    def on_start(self, state: MutableMapping[str, object]) -> None:
        self.wait = 0
        self.cooldown_counter = 0
        self.best = None

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        current = state['metrics'].get(self.monitor)
        if current is None:
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0

        if self.best is None:
            self.best = current
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif self.cooldown_counter <= 0:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.model.optimizer.learning_rate)
                new_lr = max(old_lr * self.factor, self.min_lr)
                if new_lr < old_lr:
                    self.model.optimizer.learning_rate = new_lr
                    print(f"FastMLX-ReduceLROnPlateau: Reducing LR from {old_lr:.6f} to {new_lr:.6f}")
                    self.cooldown_counter = self.cooldown
                    self.wait = 0


class TerminateOnNaN(Trace):
    """Stop training when a NaN loss is encountered.

    Args:
        monitor: Key to monitor for NaN values. Defaults to 'loss'.
    """

    def __init__(self, monitor: str = "loss") -> None:
        self.monitor = monitor

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        loss = batch.get(self.monitor)
        if loss is None:
            return

        # Handle MLX arrays
        if hasattr(loss, 'item'):
            loss_val = float(loss.item())
        else:
            loss_val = float(loss)

        if math.isnan(loss_val) or math.isinf(loss_val):
            state['should_stop'] = True
            print(f"FastMLX-TerminateOnNaN: Invalid loss value {loss_val}, stopping training")


class WarmupScheduler(Trace):
    """Warmup learning rate from a small value to the target over several steps.

    Args:
        model: The model whose optimizer's learning rate will be adjusted.
        warmup_steps: Number of steps over which to warm up.
        target_lr: The target learning rate after warmup.
        start_lr: The starting learning rate. Defaults to 0.
    """

    def __init__(
        self,
        model,
        warmup_steps: int,
        target_lr: float,
        start_lr: float = 0.0
    ) -> None:
        self.model = model
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.start_lr = start_lr
        self.step: int = 0

    def on_start(self, state: MutableMapping[str, object]) -> None:
        self.step = 0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        self.step += 1
        if self.step <= self.warmup_steps:
            lr = self.start_lr + (self.target_lr - self.start_lr) * (self.step / self.warmup_steps)
            self.model.optimizer.learning_rate = lr
