"""Random crop operation for image augmentation."""

from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
import mlx.core as mx

from .op import Op


class RandomCrop(Op):
    """Randomly crop images to a target size.

    If the input image is smaller than the target size, the image is padded
    with zeros before cropping.

    Args:
        inputs: Input key name.
        outputs: Output key name.
        height: Target crop height.
        width: Target crop width.
        pad_if_needed: If True, pad images smaller than crop size. Default True.

    Example:
        >>> op = RandomCrop("x", "x", height=224, width=224)
        >>> cropped = op.forward(img, {})  # Always (224, 224, C) or (B, 224, 224, C)
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        height: int,
        width: int,
        pad_if_needed: bool = True
    ) -> None:
        super().__init__(inputs, outputs)
        self.height = height
        self.width = width
        self.pad_if_needed = pad_if_needed

    def _pad_if_needed(self, x: np.ndarray, h: int, w: int) -> np.ndarray:
        """Pad image if it's smaller than crop size."""
        pad_h = max(0, self.height - h)
        pad_w = max(0, self.width - w)

        if pad_h > 0 or pad_w > 0:
            if x.ndim == 3:
                # (H, W, C)
                padding = ((0, pad_h), (0, pad_w), (0, 0))
            else:
                # (B, H, W, C)
                padding = ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
            x = np.pad(x, padding, mode='constant', constant_values=0)

        return x

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        x = np.array(data)

        if x.ndim == 4:
            # Batched: (B, H, W, C)
            batch, h, w, c = x.shape

            # Pad if needed
            if self.pad_if_needed and (h < self.height or w < self.width):
                x = self._pad_if_needed(x, h, w)
                h, w = x.shape[1], x.shape[2]

            # Check dimensions
            if h < self.height or w < self.width:
                raise ValueError(
                    f"RandomCrop: Image size ({h}, {w}) is smaller than crop size "
                    f"({self.height}, {self.width}). Enable pad_if_needed=True."
                )

            result = np.empty((batch, self.height, self.width, c), dtype=x.dtype)
            for i in range(batch):
                top = np.random.randint(0, h - self.height + 1)
                left = np.random.randint(0, w - self.width + 1)
                result[i] = x[i, top:top + self.height, left:left + self.width, :]

        elif x.ndim == 3:
            # Single image: (H, W, C)
            h, w, c = x.shape

            # Pad if needed
            if self.pad_if_needed and (h < self.height or w < self.width):
                x = self._pad_if_needed(x, h, w)
                h, w = x.shape[0], x.shape[1]

            # Check dimensions
            if h < self.height or w < self.width:
                raise ValueError(
                    f"RandomCrop: Image size ({h}, {w}) is smaller than crop size "
                    f"({self.height}, {self.width}). Enable pad_if_needed=True."
                )

            top = np.random.randint(0, h - self.height + 1)
            left = np.random.randint(0, w - self.width + 1)
            result = x[top:top + self.height, left:left + self.width, :]

        elif x.ndim == 2:
            # Grayscale without channel dim: (H, W)
            h, w = x.shape

            if self.pad_if_needed and (h < self.height or w < self.width):
                pad_h = max(0, self.height - h)
                pad_w = max(0, self.width - w)
                x = np.pad(x, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                h, w = x.shape

            if h < self.height or w < self.width:
                raise ValueError(
                    f"RandomCrop: Image size ({h}, {w}) is smaller than crop size "
                    f"({self.height}, {self.width}). Enable pad_if_needed=True."
                )

            top = np.random.randint(0, h - self.height + 1)
            left = np.random.randint(0, w - self.width + 1)
            result = x[top:top + self.height, left:left + self.width]

        else:
            raise ValueError(
                f"RandomCrop: Expected 2D, 3D, or 4D input, got {x.ndim}D"
            )

        return mx.array(result)
