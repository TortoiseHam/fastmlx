from __future__ import annotations

from typing import Any, MutableMapping

import mlx.nn as nn
import mlx.core as mx

from .op import Op

Array = mx.array


class ModelOp(Op):
    """Forward pass of an :class:`mlx.nn.Module`."""

    def __init__(self, model: nn.Module, inputs: str | list[str], outputs: str | list[str]) -> None:
        super().__init__(inputs, outputs)
        self.model: nn.Module = model

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Array:
        return self.model(data)
