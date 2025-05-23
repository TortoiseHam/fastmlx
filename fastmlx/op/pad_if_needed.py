from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
import mlx.core as mx

from .op import Op


class PadIfNeeded(Op):
    """Pad images to at least a minimum size."""

    def __init__(self, inputs: str, outputs: str,
                 min_height: int, min_width: int, value: float = 0.0) -> None:
        super().__init__(inputs, outputs)
        self.min_height = min_height
        self.min_width = min_width
        self.value = value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        x = np.array(data)
        if x.ndim == 4:
            batch, h, w, c = x.shape
        else:
            h, w, c = x.shape
            batch = None
        pad_h = max(0, self.min_height - h)
        pad_w = max(0, self.min_width - w)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_cfg = (
                (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)
            ) if batch is not None else (
                (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)
            )
            x = np.pad(x, pad_cfg, mode="constant", constant_values=self.value)
        return mx.array(x)
