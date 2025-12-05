"""Vertical flip augmentation operation."""

from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
import mlx.core as mx

from .op import Op


class VerticalFlip(Op):
    """Flip images vertically with a given probability.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the flipped image.
        prob: Probability of applying the flip.
    """

    def __init__(self, inputs: str, outputs: str, prob: float = 0.5) -> None:
        super().__init__(inputs, outputs)
        self.prob = prob

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        x = np.array(data)
        if np.random.rand() < self.prob:
            if x.ndim == 4:
                # Batch: (B, H, W, C) -> flip H axis
                x = x[:, ::-1, :, :]
            else:
                # Single: (H, W, C) -> flip H axis
                x = x[::-1, :, :]
        return mx.array(x)
