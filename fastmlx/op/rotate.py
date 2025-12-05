"""Rotation augmentation operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Tuple

import numpy as np
import mlx.core as mx

from .op import Op


class Rotate90(Op):
    """Rotate images by 90 degrees (0, 1, 2, or 3 times) with a given probability.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the rotated image.
        prob: Probability of applying rotation.
    """

    def __init__(self, inputs: str, outputs: str, prob: float = 0.5) -> None:
        super().__init__(inputs, outputs)
        self.prob = prob

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data)
        k = np.random.randint(1, 4)  # 1, 2, or 3 rotations

        if x.ndim == 4:
            # Batch: (B, H, W, C)
            x = np.rot90(x, k, axes=(1, 2))
        else:
            # Single: (H, W, C)
            x = np.rot90(x, k, axes=(0, 1))

        return mx.array(x)


class Rotate(Op):
    """Rotate images by a random angle within a range.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the rotated image.
        limit: Maximum rotation angle in degrees (symmetric around 0).
        prob: Probability of applying rotation.
        border_mode: How to fill border pixels ('constant', 'reflect', 'wrap').
        fill_value: Value to fill border pixels when border_mode='constant'.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        limit: float = 45.0,
        prob: float = 0.5,
        border_mode: str = "constant",
        fill_value: float = 0.0
    ) -> None:
        super().__init__(inputs, outputs)
        self.limit = limit
        self.prob = prob
        self.border_mode = border_mode
        self.fill_value = fill_value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        # This is a simplified rotation - for production use scipy.ndimage.rotate
        # or implement proper affine transformation
        x = np.array(data)
        angle = np.random.uniform(-self.limit, self.limit)

        try:
            from scipy.ndimage import rotate as scipy_rotate

            if x.ndim == 4:
                # Batch: rotate each image
                rotated = []
                for img in x:
                    rotated.append(scipy_rotate(
                        img, angle, axes=(0, 1), reshape=False,
                        mode=self.border_mode if self.border_mode != 'constant' else 'constant',
                        cval=self.fill_value
                    ))
                x = np.stack(rotated)
            else:
                x = scipy_rotate(
                    x, angle, axes=(0, 1), reshape=False,
                    mode=self.border_mode if self.border_mode != 'constant' else 'constant',
                    cval=self.fill_value
                )
        except ImportError:
            # Fallback to 90-degree rotations if scipy not available
            k = int(round(angle / 90)) % 4
            if k != 0:
                if x.ndim == 4:
                    x = np.rot90(x, k, axes=(1, 2))
                else:
                    x = np.rot90(x, k, axes=(0, 1))

        return mx.array(x)
