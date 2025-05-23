from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
import mlx.core as mx

from .op import Op


class HorizontalFlip(Op):
    """Flip images horizontally with a given probability."""

    def __init__(self, inputs: str, outputs: str, prob: float = 0.5) -> None:
        super().__init__(inputs, outputs)
        self.prob = prob

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        x = np.array(data)
        if np.random.rand() < self.prob:
            if x.ndim == 4:
                x = x[:, :, ::-1, :]
            else:
                x = x[:, ::-1, :]
        return mx.array(x)
