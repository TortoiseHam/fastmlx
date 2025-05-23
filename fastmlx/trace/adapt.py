"""Learning rate scheduling trace."""

from __future__ import annotations

from typing import Callable, MutableMapping


class LRScheduler:
    def __init__(self, model, lr_fn: Callable[[int], float]):
        self.model = model
        self.lr_fn = lr_fn
        self.step: int = 0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        self.step += 1
        lr = self.lr_fn(self.step)
        self.model.optimizer.learning_rate = lr
