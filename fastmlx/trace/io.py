"""Input/output related traces for model saving and logging."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import MutableMapping, Optional, List, Dict, Any

from .base import Trace

import mlx.core as mx


class BestModelSaver(Trace):
    """Save model weights when a monitored metric improves.

    Args:
        model: The model to save.
        save_dir: Directory to save the model.
        metric: Metric name to monitor (from state['metrics']).
        save_best_mode: One of 'max' or 'min'. Save when metric is highest or lowest.
    """

    def __init__(
        self,
        model,
        save_dir: str,
        metric: str = "accuracy",
        save_best_mode: str = "max"
    ) -> None:
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.metric = metric
        self.save_best_mode = save_best_mode
        self.best: Optional[float] = None

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        score = state['metrics'].get(self.metric)
        if score is None:
            return
        if self.best is None or (self.save_best_mode == "max" and score > self.best) or (
            self.save_best_mode == "min" and score < self.best
        ):
            self.best = score
            path = os.path.join(self.save_dir, "model_best.npz")
            self.model.save_weights(path)
            print(f"FastMLX-BestModelSaver: Saved model to {path}")


class ModelSaver(Trace):
    """Save model weights at specified intervals.

    Args:
        model: The model to save.
        save_dir: Directory to save the model.
        frequency: How often to save, in epochs. If 0, only save at the end.
        save_optimizer: Whether to save optimizer state as well.
    """

    def __init__(
        self,
        model,
        save_dir: str,
        frequency: int = 1,
        save_optimizer: bool = False
    ) -> None:
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.frequency = frequency
        self.save_optimizer = save_optimizer

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        epoch = state.get('epoch', 0)
        if self.frequency > 0 and (epoch + 1) % self.frequency == 0:
            self._save(epoch)

    def on_finish(self, state: MutableMapping[str, object]) -> None:
        epoch = state.get('epoch', 0)
        self._save(epoch, final=True)

    def _save(self, epoch: int, final: bool = False) -> None:
        if final:
            model_path = os.path.join(self.save_dir, "model_final.npz")
        else:
            model_path = os.path.join(self.save_dir, f"model_epoch_{epoch + 1}.npz")

        self.model.save_weights(model_path)
        print(f"FastMLX-ModelSaver: Saved model to {model_path}")

        if self.save_optimizer and hasattr(self.model, 'optimizer'):
            if final:
                opt_path = os.path.join(self.save_dir, "optimizer_final.npz")
            else:
                opt_path = os.path.join(self.save_dir, f"optimizer_epoch_{epoch + 1}.npz")
            # Save optimizer state
            opt_state = self.model.optimizer.state
            mx.savez(opt_path, **{f"opt_{k}": v for k, v in enumerate(opt_state) if isinstance(v, mx.array)})


class CSVLogger(Trace):
    """Log training metrics to a CSV file.

    Args:
        filename: Path to the CSV file.
        separator: Delimiter for the CSV file.
        append: If True, append to existing file. If False, overwrite.
    """

    def __init__(
        self,
        filename: str,
        separator: str = ",",
        append: bool = False
    ) -> None:
        self.filename = filename
        self.separator = separator
        self.append = append
        self.keys: Optional[List[str]] = None
        self.file_handle = None
        self.writer = None

    def on_start(self, state: MutableMapping[str, object]) -> None:
        mode = 'a' if self.append else 'w'
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filename) if os.path.dirname(self.filename) else '.', exist_ok=True)
        self.file_handle = open(self.filename, mode, newline='')
        self.writer = csv.writer(self.file_handle, delimiter=self.separator)

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        metrics = state.get('metrics', {})
        epoch = state.get('epoch', 0)
        mode = state.get('mode', 'train')

        row_dict: Dict[str, Any] = {'epoch': epoch, 'mode': mode}
        row_dict.update(metrics)

        # Write header on first epoch
        if self.keys is None:
            self.keys = list(row_dict.keys())
            if not self.append:
                self.writer.writerow(self.keys)

        # Write values
        row = [row_dict.get(k, '') for k in self.keys]
        self.writer.writerow(row)
        self.file_handle.flush()

    def on_finish(self, state: MutableMapping[str, object]) -> None:
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None


class ProgressLogger(Trace):
    """Print training progress to console.

    Args:
        log_frequency: How often to log (in batches). If 0, log only at epoch end.
    """

    def __init__(self, log_frequency: int = 0) -> None:
        self.log_frequency = log_frequency
        self.batch_count: int = 0

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.batch_count = 0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        self.batch_count += 1
        if self.log_frequency > 0 and self.batch_count % self.log_frequency == 0:
            epoch = state.get('epoch', 0)
            mode = state.get('mode', 'train')
            loss = batch.get('loss') or batch.get('ce')
            if loss is not None:
                if hasattr(loss, 'item'):
                    loss = float(loss.item())
                print(f"Epoch {epoch} [{mode}] Batch {self.batch_count}: loss={loss:.4f}")


class Timer(Trace):
    """Track and report training time.

    Records time for each epoch and total training time.
    """

    def __init__(self) -> None:
        self.start_time: Optional[datetime] = None
        self.epoch_start: Optional[datetime] = None

    def on_start(self, state: MutableMapping[str, object]) -> None:
        self.start_time = datetime.now()
        print(f"FastMLX-Timer: Training started at {self.start_time}")

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.epoch_start = datetime.now()

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        if self.epoch_start:
            epoch_time = datetime.now() - self.epoch_start
            epoch = state.get('epoch', 0)
            mode = state.get('mode', 'train')
            state['metrics']['epoch_time'] = epoch_time.total_seconds()
            print(f"FastMLX-Timer: Epoch {epoch} [{mode}] completed in {epoch_time}")

    def on_finish(self, state: MutableMapping[str, object]) -> None:
        if self.start_time:
            total_time = datetime.now() - self.start_time
            print(f"FastMLX-Timer: Training completed in {total_time}")
