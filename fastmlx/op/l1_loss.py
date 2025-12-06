"""L1 (Mean Absolute Error) loss operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

import mlx.core as mx

from .op import LossOp


class L1Loss(LossOp):
    """Compute the L1 (Mean Absolute Error) loss.

    This loss computes the mean of absolute differences between predictions and targets.
    More robust to outliers than MSE.

    Args:
        inputs: A tuple of (prediction_key, target_key).
        outputs: The key to store the computed loss.
        reduction: How to reduce the loss. One of 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        inputs: Sequence[str],
        outputs: str,
        reduction: str = "mean"
    ) -> None:
        super().__init__(inputs, outputs)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        self.reduction = reduction

    def forward(self, data: Sequence[mx.array], state: MutableMapping[str, Any]) -> mx.array:
        y_pred, y_true = data
        abs_diff = mx.abs(y_pred - y_true)

        if self.reduction == "mean":
            return mx.mean(abs_diff)
        elif self.reduction == "sum":
            return mx.sum(abs_diff)
        else:
            return abs_diff
