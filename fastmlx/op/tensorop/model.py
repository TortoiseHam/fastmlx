"""Operations wrapping MLX models."""

from __future__ import annotations

from typing import Any, MutableMapping

import mlx.nn as nn
import mlx.core as mx
Array = mx.array
from ..op import Op


class ModelOp(Op):
    def __init__(self, model: nn.Module, inputs: str | list[str], outputs: str | list[str]) -> None:
        super().__init__(inputs, outputs)
        self.model: nn.Module = model

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        return self.model(data)


class UpdateOp(Op):
    def __init__(self, model: nn.Module, loss_name: str) -> None:
        super().__init__(inputs=loss_name, outputs=None)
        self.model: nn.Module = model

    def forward(self, data: Array, state: MutableMapping[str, Any]) -> None:
        batch = state.get("batch", {})
        x = batch.get("x")
        y = batch.get("y")
        if x is None or y is None:
            return None

        if not isinstance(x, mx.array):
            x = mx.array(x)
        if not isinstance(y, mx.array):
            y = mx.array(y)

        def loss_fn(x_data, y_data):
            logits = self.model(x_data)
            loss = nn.losses.cross_entropy(logits, y_data)
            return mx.mean(loss)

        value, grads = nn.value_and_grad(self.model, loss_fn)(x, y)
        batch[self.inputs[0]] = value
        self.model.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())
        return None
