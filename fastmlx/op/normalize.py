"""Normalize operation for image data."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence, Union

import mlx.core as mx
import numpy as np

from .op import Op


class Normalize(Op):
    """Normalize image data using mean and standard deviation.

    Performs the transformation: output = (input / max_pixel_value - mean) / std

    This op works with both numpy arrays (sample ops) and MLX arrays (batch ops).

    Args:
        inputs: Input key name.
        outputs: Output key name.
        mean: Mean values for each channel. Can be a single float or sequence.
        std: Standard deviation values for each channel. Can be a single float or sequence.
            Values of 0 are replaced with 1.0 to avoid division by zero.
        max_pixel_value: Maximum pixel value for normalization (default 255.0 for uint8).

    Example:
        >>> # ImageNet normalization
        >>> op = Normalize("x", "x",
        ...                mean=(0.485, 0.456, 0.406),
        ...                std=(0.229, 0.224, 0.225))
        >>> normalized = op.forward(img, {})
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        mean: Union[float, Sequence[float]] = (0.0, 0.0, 0.0),
        std: Union[float, Sequence[float]] = (1.0, 1.0, 1.0),
        max_pixel_value: float = 255.0
    ) -> None:
        super().__init__(inputs, outputs)

        # Handle scalar inputs
        if isinstance(mean, (int, float)):
            mean = [float(mean)]
        if isinstance(std, (int, float)):
            std = [float(std)]

        # Store as numpy arrays for sample ops compatibility
        self.mean = np.array(list(mean), dtype=np.float32)

        # Replace zeros in std with 1.0 to avoid division by zero
        std_list = list(std)
        std_safe = [s if s != 0 else 1.0 for s in std_list]
        self.std = np.array(std_safe, dtype=np.float32)

        self.max_pixel_value = max_pixel_value

        # Track if any std values were zero (for warning)
        self._had_zero_std = any(s == 0 for s in std_list)

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Any:
        # Handle both numpy arrays (sample ops) and MLX arrays (batch ops)
        is_numpy = isinstance(data, np.ndarray)

        if is_numpy:
            x = data.astype(np.float32)
            mean = self.mean
            std = self.std
        else:
            x = data.astype(mx.float32)
            mean = mx.array(self.mean)
            std = mx.array(self.std)

        # Scale to [0, 1] if needed
        if self.max_pixel_value != 1.0:
            x = x / self.max_pixel_value

        # Normalize
        # Handle broadcasting for different input shapes
        # Input could be (H, W, C), (B, H, W, C), or even (C,) for 1D
        return (x - mean) / std
