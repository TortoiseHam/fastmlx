from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
import mlx.core as mx

from .op import Op


class Onehot(Op):
    """Convert integer labels to one-hot vectors."""

    def __init__(self, inputs: str, outputs: str, num_classes: int,
                 label_smoothing: float = 0.0) -> None:
        super().__init__(inputs, outputs)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        y = np.array(data).astype(int)
        oh = np.eye(self.num_classes, dtype=np.float32)[y]
        if self.label_smoothing:
            smooth = self.label_smoothing / self.num_classes
            oh = np.where(oh == 1.0, 1.0 - self.label_smoothing + smooth, smooth)
        return mx.array(oh)
