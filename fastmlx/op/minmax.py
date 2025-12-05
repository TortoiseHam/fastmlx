"""Minmax scaling operation for image data."""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

import mlx.core as mx

from .op import Op


class Minmax(Op):
    """Scale data to [0, 1] range using min-max normalization.

    By default, assumes uint8 images with range [0, 255]. Can also handle
    data that is already in float format.

    Args:
        inputs: Input key name.
        outputs: Output key name.
        input_min: Minimum value of input data. If None, auto-detects:
            - uint8: uses 0
            - float already in [0,1]: no scaling
            - other: uses actual min
        input_max: Maximum value of input data. If None, auto-detects:
            - uint8: uses 255
            - float already in [0,1]: no scaling
            - other: uses actual max

    Example:
        >>> op = Minmax("x", "x")
        >>> img = mx.array([0, 127, 255], dtype=mx.uint8)
        >>> scaled = op.forward(img, {})  # [0.0, 0.498, 1.0]
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        input_min: Optional[float] = None,
        input_max: Optional[float] = None
    ) -> None:
        super().__init__(inputs, outputs)
        self.input_min = input_min
        self.input_max = input_max

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        # Convert to float32 for processing
        x = data.astype(mx.float32)

        # Determine input range
        if self.input_min is not None and self.input_max is not None:
            # Use specified range
            min_val = self.input_min
            max_val = self.input_max
        elif data.dtype == mx.uint8:
            # Standard uint8 image
            min_val = 0.0
            max_val = 255.0
        elif data.dtype in (mx.uint16, mx.uint32):
            # Higher bit depth images
            min_val = 0.0
            max_val = float(2 ** (16 if data.dtype == mx.uint16 else 32) - 1)
        elif data.dtype in (mx.float16, mx.float32, mx.bfloat16):
            # Float data - check if already normalized
            actual_min = float(mx.min(x).item())
            actual_max = float(mx.max(x).item())

            # If already in [0, 1] range, return as-is
            if actual_min >= -0.01 and actual_max <= 1.01:
                return x

            # Otherwise use actual range
            min_val = actual_min
            max_val = actual_max
        else:
            # Default: assume 0-255 for backward compatibility
            min_val = 0.0
            max_val = 255.0

        # Avoid division by zero
        range_val = max_val - min_val
        if range_val == 0:
            return mx.zeros_like(x)

        return (x - min_val) / range_val
