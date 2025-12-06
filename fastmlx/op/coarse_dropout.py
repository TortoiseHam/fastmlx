from __future__ import annotations

from typing import Any, MutableMapping

import mlx.core as mx
import numpy as np

from .op import Op


class CoarseDropout(Op):
    """Randomly drop rectangular patches from an image.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the augmented image.
        max_holes: Maximum number of rectangular patches to drop.
        max_height: Maximum height of each patch.
        max_width: Maximum width of each patch.
        fill_value: Value to fill the dropped regions with.
        prob: Probability of applying the augmentation.
    """

    def __init__(self, inputs: str, outputs: str, max_holes: int = 1,
                 max_height: int = 8, max_width: int = 8, fill_value: float = 0.0,
                 prob: float = 0.5) -> None:
        super().__init__(inputs, outputs)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.fill_value = fill_value
        self.prob = prob

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data
        x = np.array(data)
        if x.ndim == 4:
            batch, h, w, c = x.shape
            for i in range(batch):
                for _ in range(self.max_holes):
                    hole_h = np.random.randint(1, self.max_height + 1)
                    hole_w = np.random.randint(1, self.max_width + 1)
                    top = np.random.randint(0, max(1, h - hole_h + 1))
                    left = np.random.randint(0, max(1, w - hole_w + 1))
                    x[i, top:top + hole_h, left:left + hole_w, :] = self.fill_value
        else:
            h, w, c = x.shape
            for _ in range(self.max_holes):
                hole_h = np.random.randint(1, self.max_height + 1)
                hole_w = np.random.randint(1, self.max_width + 1)
                top = np.random.randint(0, max(1, h - hole_h + 1))
                left = np.random.randint(0, max(1, w - hole_w + 1))
                x[top:top + hole_h, left:left + hole_w, :] = self.fill_value
        return mx.array(x)
