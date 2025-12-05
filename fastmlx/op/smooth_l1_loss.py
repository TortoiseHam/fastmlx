"""Smooth L1 (Huber) Loss operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

import mlx.core as mx

from .op import Op


class SmoothL1Loss(Op):
    """Compute the Smooth L1 (Huber) loss.

    Smooth L1 loss is less sensitive to outliers than MSE and behaves like L1 for
    large errors. Commonly used in object detection (e.g., Fast R-CNN).

    L(x) = 0.5 * x^2 / beta,          if |x| < beta
           |x| - 0.5 * beta,          otherwise

    Args:
        inputs: A tuple of (prediction_key, target_key).
        outputs: The key to store the computed loss.
        beta: Threshold at which to change from L2 to L1 behavior.
        reduction: How to reduce the loss. One of 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        inputs: Sequence[str],
        outputs: str,
        beta: float = 1.0,
        reduction: str = "mean"
    ) -> None:
        super().__init__(inputs, outputs)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = beta
        self.reduction = reduction

    def forward(self, data: Sequence[mx.array], state: MutableMapping[str, Any]) -> mx.array:
        y_pred, y_true = data
        diff = y_pred - y_true
        abs_diff = mx.abs(diff)

        # Smooth L1: quadratic for small errors, linear for large errors
        loss = mx.where(
            abs_diff < self.beta,
            0.5 * mx.square(diff) / self.beta,
            abs_diff - 0.5 * self.beta
        )

        if self.reduction == "mean":
            return mx.mean(loss)
        elif self.reduction == "sum":
            return mx.sum(loss)
        else:
            return loss
