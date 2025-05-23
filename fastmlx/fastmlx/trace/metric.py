"""Metric traces used during training."""

from __future__ import annotations

from typing import MutableMapping

import numpy as np


class Accuracy:
    def __init__(self, true_key: str = "y", pred_key: str = "y_pred") -> None:
        self.true_key = true_key
        self.pred_key = pred_key
        self.correct: int = 0
        self.total: int = 0

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.correct = 0
        self.total = 0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]
        pred = y_pred.argmax(axis=-1)
        self.correct += int(np.sum(pred == y))
        self.total += len(y)

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        state['metrics']["accuracy"] = self.correct / max(1, self.total)
