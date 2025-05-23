"""Metric traces used during training."""

from __future__ import annotations

from typing import MutableMapping

from .base import Trace

import mlx.core as mx


class Accuracy(Trace):
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
        if not isinstance(y_pred, mx.array):
            y_pred = mx.array(y_pred)
        if not isinstance(y, mx.array):
            y = mx.array(y)
        pred = mx.argmax(y_pred, axis=-1)
        self.correct += int(mx.sum(pred == y).item())
        self.total += y.shape[0]

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        state['metrics']["accuracy"] = self.correct / max(1, self.total)


class LossMonitor(Trace):
    """Track the mean of a given loss value over an epoch."""

    def __init__(self, loss_key: str = "ce") -> None:
        self.loss_key = loss_key
        self.total: float = 0.0
        self.count: int = 0

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.total = 0.0
        self.count = 0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        loss = batch.get(self.loss_key)
        if loss is None:
            return
        if isinstance(loss, mx.array):
            loss = float(loss.item())
        self.total += float(loss)
        self.count += 1

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        if self.count:
            state['metrics'][self.loss_key] = self.total / self.count
