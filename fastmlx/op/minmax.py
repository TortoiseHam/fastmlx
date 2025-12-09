"""Minmax scaling operation for image data."""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

import mlx.core as mx
import numpy as np

from .op import Op


class Minmax(Op):
    """Scale data to [0, 1] range using min-max normalization.

    By default, assumes uint8 images with range [0, 255]. Can also handle
    data that is already in float format.

    This op works with both numpy arrays (sample ops) and MLX arrays (batch ops).

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
        >>> img = np.array([0, 127, 255], dtype=np.uint8)
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

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Any:
        # Handle both numpy arrays (sample ops) and MLX arrays (batch ops)
        is_numpy = isinstance(data, np.ndarray)

        # Convert to float32 for processing
        if is_numpy:
            x = data.astype(np.float32)
        else:
            x = data.astype(mx.float32)

        # Determine input range
        if self.input_min is not None and self.input_max is not None:
            # Use specified range
            min_val = self.input_min
            max_val = self.input_max
        elif is_numpy:
            # Handle numpy dtypes
            if data.dtype == np.uint8:
                min_val = 0.0
                max_val = 255.0
            elif data.dtype in (np.uint16, np.uint32):
                min_val = 0.0
                max_val = float(2 ** (16 if data.dtype == np.uint16 else 32) - 1)
            elif data.dtype in (np.float16, np.float32, np.float64):
                actual_min = float(np.min(x))
                actual_max = float(np.max(x))
                if actual_min >= -0.01 and actual_max <= 1.01:
                    return x
                min_val = actual_min
                max_val = actual_max
            else:
                min_val = 0.0
                max_val = 255.0
        else:
            # Handle MLX dtypes
            if data.dtype == mx.uint8:
                min_val = 0.0
                max_val = 255.0
            elif data.dtype in (mx.uint16, mx.uint32):
                min_val = 0.0
                max_val = float(2 ** (16 if data.dtype == mx.uint16 else 32) - 1)
            elif data.dtype in (mx.float16, mx.float32, mx.bfloat16):
                actual_min = float(mx.min(x).item())
                actual_max = float(mx.max(x).item())
                if actual_min >= -0.01 and actual_max <= 1.01:
                    return x
                min_val = actual_min
                max_val = actual_max
            else:
                min_val = 0.0
                max_val = 255.0

        # Avoid division by zero
        range_val = max_val - min_val
        if range_val == 0:
            if is_numpy:
                return np.zeros_like(x)
            return mx.zeros_like(x)

        return (x - min_val) / range_val
