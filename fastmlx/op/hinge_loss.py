"""Hinge Loss operation for SVMs and margin-based classification."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

import mlx.core as mx

from .op import LossOp


class HingeLoss(LossOp):
    """Compute the hinge loss for margin-based classification.

    Hinge loss is commonly used for training SVMs and other maximum-margin classifiers.

    For binary classification:
        L = max(0, 1 - y * y_pred)

    For multi-class (one-vs-all):
        L = max(0, 1 - (y_pred[true_class] - max(y_pred[other_classes])))

    Args:
        inputs: A tuple of (prediction_key, target_key).
        outputs: The key to store the computed loss.
        reduction: How to reduce the loss. One of 'mean', 'sum', or 'none'.

    Example:
        >>> op = HingeLoss(["y_pred", "y"], "loss")
        >>> y_pred = mx.array([2.0, -1.0])  # SVM outputs
        >>> y_true = mx.array([1.0, -1.0])  # +1 or -1 labels
        >>> loss = op.forward([y_pred, y_true], {})
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

        # Handle multi-class case
        if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
            num_classes = y_pred.shape[-1]

            # Get true class indices
            if y_true.ndim > 1:
                y_true_idx = mx.argmax(y_true, axis=-1)
            else:
                y_true_idx = y_true.astype(mx.int32)

            # Get score of true class using take_along_axis
            true_scores = mx.take_along_axis(
                y_pred,
                y_true_idx.reshape(-1, 1),
                axis=1
            ).squeeze(-1)

            # Create mask to exclude true class (vectorized)
            # Create one-hot mask for true class
            eye = mx.eye(num_classes)
            true_class_mask = eye[y_true_idx]  # Shape: (batch_size, num_classes)

            # Mask out true class scores with large negative value
            masked_scores = y_pred - true_class_mask * 1e9

            # Get max of other classes
            other_max = mx.max(masked_scores, axis=-1)

            # Multi-class hinge loss
            hinge = mx.maximum(0.0, 1.0 - (true_scores - other_max))
        else:
            # Binary case: y_true should be +1 or -1
            # If y_true is 0/1, convert to -1/+1
            y_true_signed = mx.where(y_true > 0.5, 1.0, -1.0)
            y_pred_squeezed = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
            hinge = mx.maximum(0.0, 1.0 - y_true_signed * y_pred_squeezed)

        if self.reduction == "mean":
            return mx.mean(hinge)
        elif self.reduction == "sum":
            return mx.sum(hinge)
        else:
            return hinge
