"""Brightness and contrast augmentation operations."""

from __future__ import annotations

from typing import Any, MutableMapping, Tuple, Union

import mlx.core as mx
import numpy as np

from .op import Op


class Brightness(Op):
    """Adjust image brightness.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the adjusted image.
        limit: Range for brightness adjustment factor.
              Brightness is adjusted as: img + limit * max_value
        prob: Probability of applying brightness adjustment.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        limit: Union[float, Tuple[float, float]] = (-0.2, 0.2),
        prob: float = 0.5
    ) -> None:
        super().__init__(inputs, outputs)
        if isinstance(limit, (int, float)):
            self.limit = (-abs(limit), abs(limit))
        else:
            self.limit = limit
        self.prob = prob

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data).astype(np.float32)

        # Determine max value
        max_val = 255.0 if x.max() > 1.0 else 1.0

        # Random brightness factor
        factor = np.random.uniform(self.limit[0], self.limit[1])
        x = x + factor * max_val

        # Clip to valid range
        x = np.clip(x, 0, max_val)

        return mx.array(x)


class Contrast(Op):
    """Adjust image contrast.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the adjusted image.
        limit: Range for contrast adjustment factor.
              Contrast is adjusted as: (img - mean) * factor + mean
        prob: Probability of applying contrast adjustment.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        limit: Union[float, Tuple[float, float]] = (0.8, 1.2),
        prob: float = 0.5
    ) -> None:
        super().__init__(inputs, outputs)
        if isinstance(limit, (int, float)):
            self.limit = (1.0 - abs(limit - 1.0), 1.0 + abs(limit - 1.0))
        else:
            self.limit = limit
        self.prob = prob

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data).astype(np.float32)

        # Determine max value
        max_val = 255.0 if x.max() > 1.0 else 1.0

        # Random contrast factor
        factor = np.random.uniform(self.limit[0], self.limit[1])

        # Adjust contrast
        mean = x.mean()
        x = (x - mean) * factor + mean

        # Clip to valid range
        x = np.clip(x, 0, max_val)

        return mx.array(x)


class RandomBrightnessContrast(Op):
    """Randomly adjust both brightness and contrast.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the adjusted image.
        brightness_limit: Range for brightness adjustment.
        contrast_limit: Range for contrast adjustment.
        prob: Probability of applying adjustments.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        brightness_limit: Union[float, Tuple[float, float]] = (-0.2, 0.2),
        contrast_limit: Union[float, Tuple[float, float]] = (0.8, 1.2),
        prob: float = 0.5
    ) -> None:
        super().__init__(inputs, outputs)
        if isinstance(brightness_limit, (int, float)):
            self.brightness_limit = (-abs(brightness_limit), abs(brightness_limit))
        else:
            self.brightness_limit = brightness_limit
        if isinstance(contrast_limit, (int, float)):
            self.contrast_limit = (1.0 - abs(contrast_limit - 1.0), 1.0 + abs(contrast_limit - 1.0))
        else:
            self.contrast_limit = contrast_limit
        self.prob = prob

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data).astype(np.float32)

        # Determine max value
        max_val = 255.0 if x.max() > 1.0 else 1.0

        # Apply contrast first
        contrast = np.random.uniform(self.contrast_limit[0], self.contrast_limit[1])
        mean = x.mean()
        x = (x - mean) * contrast + mean

        # Then brightness
        brightness = np.random.uniform(self.brightness_limit[0], self.brightness_limit[1])
        x = x + brightness * max_val

        # Clip to valid range
        x = np.clip(x, 0, max_val)

        return mx.array(x)
