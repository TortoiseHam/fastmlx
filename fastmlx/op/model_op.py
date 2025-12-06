from __future__ import annotations

from typing import Any, MutableMapping

import mlx.core as mx
import mlx.nn as nn

from .op import Op


class ModelOp(Op):
    """Forward pass of an :class:`mlx.nn.Module`."""

    def __init__(self, model: nn.Module, inputs: str | list[str], outputs: str | list[str]) -> None:
        super().__init__(inputs, outputs)
        self.model: nn.Module = model

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> mx.array:
        return self.model(data)
