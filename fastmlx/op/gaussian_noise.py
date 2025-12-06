"""Gaussian noise augmentation operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Tuple, Union

import mlx.core as mx
import numpy as np

from .op import Op


class GaussianNoise(Op):
    """Add Gaussian noise to images.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the noisy image.
        var_limit: Range for noise variance. Can be float or tuple (min, max).
        mean: Mean of the Gaussian noise.
        prob: Probability of applying noise.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        var_limit: Union[float, Tuple[float, float]] = (10.0, 50.0),
        mean: float = 0.0,
        prob: float = 0.5
    ) -> None:
        super().__init__(inputs, outputs)
        if isinstance(var_limit, (int, float)):
            self.var_limit = (0.0, float(var_limit))
        else:
            self.var_limit = var_limit
        self.mean = mean
        self.prob = prob

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data).astype(np.float32)

        # Random variance
        var = np.random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5

        # Add Gaussian noise
        noise = np.random.normal(self.mean, sigma, x.shape).astype(np.float32)
        x = x + noise

        # Clip to valid range (assuming 0-255 or 0-1)
        if x.max() > 1.0:
            x = np.clip(x, 0, 255)
        else:
            x = np.clip(x, 0, 1)

        return mx.array(x)
