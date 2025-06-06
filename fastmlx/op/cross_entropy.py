from __future__ import annotations

from typing import Any, MutableMapping, Sequence

import mlx.nn as nn
import mlx.core as mx

from .op import Op


class CrossEntropy(Op):
    """Compute the mean cross entropy loss."""

    def __init__(self, inputs: Sequence[str], outputs: str) -> None:
        super().__init__(inputs, outputs)

    def forward(self, data: Sequence[mx.array], state: MutableMapping[str, Any]) -> mx.array:
        y_pred, y_true = data
        loss = nn.losses.cross_entropy(y_pred, y_true)
        return mx.mean(loss)
