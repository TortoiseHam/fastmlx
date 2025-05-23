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

    def forward(self, data: Array, state: MutableMapping[str, Any]) -> Array:
        return self.model(data)


class UpdateOp(Op):
    def __init__(self, model: nn.Module, loss_name: str) -> None:
        super().__init__(inputs=loss_name, outputs=None)
        self.model: nn.Module = model

    def forward(self, data: Array, state: MutableMapping[str, Any]) -> None:
        loss = data
        grads = mx.grad(loss, self.model.trainable_parameters())
        self.model.optimizer.update(self.model, grads)
        return None
