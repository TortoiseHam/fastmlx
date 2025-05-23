from __future__ import annotations

from typing import Any, MutableMapping

import mlx.core as mx

from .op import Op


class ExpandDims(Op):
    """Insert a new axis into an array."""

    def __init__(self, inputs: str, outputs: str, axis: int = 0) -> None:
        super().__init__(inputs, outputs)
        self.axis = axis

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        return mx.expand_dims(data, axis=self.axis)
