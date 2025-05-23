from __future__ import annotations

from typing import Any, MutableMapping

import mlx.core as mx

from .op import Op


class Minmax(Op):
    """Scale image data to [0,1] range."""

    def __init__(self, inputs: str, outputs: str) -> None:
        super().__init__(inputs, outputs)

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        return data.astype(mx.float32) / 255.0
