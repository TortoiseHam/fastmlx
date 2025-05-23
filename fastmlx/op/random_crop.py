from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
import mlx.core as mx

from .op import Op


class RandomCrop(Op):
    """Randomly crop images to a target size."""

    def __init__(self, inputs: str, outputs: str, height: int, width: int) -> None:
        super().__init__(inputs, outputs)
        self.height = height
        self.width = width

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        x = np.array(data)
        if x.ndim == 4:
            batch, h, w, c = x.shape
            result = np.empty((batch, self.height, self.width, c), dtype=x.dtype)
            for i in range(batch):
                top = np.random.randint(0, h - self.height + 1)
                left = np.random.randint(0, w - self.width + 1)
                result[i] = x[i, top:top + self.height, left:left + self.width, :]
        else:
            h, w, c = x.shape
            top = np.random.randint(0, h - self.height + 1)
            left = np.random.randint(0, w - self.width + 1)
            result = x[top:top + self.height, left:left + self.width, :]
        return mx.array(result)
