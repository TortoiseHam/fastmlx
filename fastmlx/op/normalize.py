from __future__ import annotations

from typing import Any, MutableMapping, Sequence

import mlx.core as mx

from .op import Op


class Normalize(Op):
    """Normalize image data using mean and std."""

    def __init__(self, inputs: str, outputs: str,
                 mean: Sequence[float] = (0.0, 0.0, 0.0),
                 std: Sequence[float] = (1.0, 1.0, 1.0),
                 max_pixel_value: float = 255.0) -> None:
        super().__init__(inputs, outputs)
        self.mean = mx.array(mean, dtype=mx.float32)
        self.std = mx.array(std, dtype=mx.float32)
        self.max_pixel_value = max_pixel_value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        x = data.astype(mx.float32) / self.max_pixel_value
        return (x - self.mean) / self.std
