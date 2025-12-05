"""Dice Loss operation for segmentation tasks."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

import mlx.core as mx

from .op import Op


class DiceLoss(Op):
    """Compute the Dice loss for segmentation tasks.

    Dice loss is based on the Sorensen-Dice coefficient, measuring overlap between
    predicted and target segmentation masks. Particularly useful for imbalanced
    segmentation tasks.

    Dice = 2 * |X intersect Y| / (|X| + |Y|)
    DiceLoss = 1 - Dice

    Args:
        inputs: A tuple of (prediction_key, target_key).
        outputs: The key to store the computed loss.
        smooth: Smoothing factor to avoid division by zero.
        reduction: How to reduce the loss. One of 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        inputs: Sequence[str],
        outputs: str,
        smooth: float = 1.0,
        reduction: str = "mean"
    ) -> None:
        super().__init__(inputs, outputs)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, data: Sequence[mx.array], state: MutableMapping[str, Any]) -> mx.array:
        y_pred, y_true = data

        # Apply sigmoid if logits are provided (values outside [0, 1])
        if mx.any(y_pred < 0) or mx.any(y_pred > 1):
            y_pred = mx.sigmoid(y_pred)

        # Flatten predictions and targets for each sample in batch
        batch_size = y_pred.shape[0]
        y_pred_flat = y_pred.reshape(batch_size, -1)
        y_true_flat = y_true.reshape(batch_size, -1)

        # Compute intersection and union
        intersection = mx.sum(y_pred_flat * y_true_flat, axis=1)
        union = mx.sum(y_pred_flat, axis=1) + mx.sum(y_true_flat, axis=1)

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss is 1 - Dice
        dice_loss = 1.0 - dice

        if self.reduction == "mean":
            return mx.mean(dice_loss)
        elif self.reduction == "sum":
            return mx.sum(dice_loss)
        else:
            return dice_loss
