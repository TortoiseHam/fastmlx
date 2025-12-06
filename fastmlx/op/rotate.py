"""Rotation augmentation operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

import mlx.core as mx
import numpy as np

from .op import Op


class Rotate90(Op):
    """Rotate images by 90 degrees (0, 1, 2, or 3 times).

    Args:
        inputs: Input key for the image.
        outputs: Output key for the rotated image.
        k: Number of 90-degree rotations. If None, randomly chooses 1, 2, or 3.
        prob: Probability of applying rotation (only used when k is None).

    Example:
        >>> # Deterministic 90-degree rotation
        >>> op = Rotate90("x", "x", k=1)
        >>> rotated = op.forward(img, {})
        >>>
        >>> # Random rotation with 50% probability
        >>> op = Rotate90("x", "x", prob=0.5)
        >>> maybe_rotated = op.forward(img, {})
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        k: Optional[int] = None,
        prob: float = 0.5
    ) -> None:
        super().__init__(inputs, outputs)
        self.k = k
        self.prob = prob

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        # Determine rotation amount
        if self.k is not None:
            # Deterministic rotation
            k = self.k % 4
            if k == 0:
                return data
        else:
            # Probabilistic rotation
            if np.random.rand() >= self.prob:
                return data
            k = np.random.randint(1, 4)  # 1, 2, or 3 rotations

        x = np.array(data)

        if x.ndim == 4:
            # Batch: (B, H, W, C)
            x = np.rot90(x, k, axes=(1, 2))
        elif x.ndim == 3:
            # Single: (H, W, C)
            x = np.rot90(x, k, axes=(0, 1))
        elif x.ndim == 2:
            # Grayscale: (H, W)
            x = np.rot90(x, k)
        else:
            raise ValueError(f"Rotate90: Expected 2D, 3D, or 4D input, got {x.ndim}D")

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

    Example:
        >>> op = Rotate("x", "x", limit=30, prob=0.5)
        >>> rotated = op.forward(img, {})
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
            elif x.ndim == 3:
                x = scipy_rotate(
                    x, angle, axes=(0, 1), reshape=False,
                    mode=self.border_mode if self.border_mode != 'constant' else 'constant',
                    cval=self.fill_value
                )
            elif x.ndim == 2:
                x = scipy_rotate(
                    x, angle, reshape=False,
                    mode=self.border_mode if self.border_mode != 'constant' else 'constant',
                    cval=self.fill_value
                )
            else:
                raise ValueError(f"Rotate: Expected 2D, 3D, or 4D input, got {x.ndim}D")
        except ImportError:
            # Fallback to 90-degree rotations if scipy not available
            k = int(round(angle / 90)) % 4
            if k != 0:
                if x.ndim == 4:
                    x = np.rot90(x, k, axes=(1, 2))
                elif x.ndim == 3:
                    x = np.rot90(x, k, axes=(0, 1))
                elif x.ndim == 2:
                    x = np.rot90(x, k)

        return mx.array(x)
