"""Focal Loss operation for handling class imbalance."""

from __future__ import annotations

from typing import Any, MutableMapping, Optional, Sequence

import mlx.core as mx

from .op import Op


class FocalLoss(Op):
    """Compute the focal loss for classification with class imbalance.

    Focal loss down-weights well-classified examples and focuses on hard examples.
    Originally proposed for dense object detection.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        inputs: A tuple of (prediction_key, target_key).
        outputs: The key to store the computed loss.
        alpha: Weighting factor for the rare class. If None, no weighting is applied.
        gamma: Focusing parameter. Higher values focus more on hard examples.
        reduction: How to reduce the loss. One of 'mean', 'sum', or 'none'.

    Example:
        >>> op = FocalLoss(["y_pred", "y"], "loss", gamma=2.0)
        >>> y_pred = mx.array([[0.9, 0.1], [0.2, 0.8]])
        >>> y_true = mx.array([0, 1])
        >>> loss = op.forward([y_pred, y_true], {})

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        inputs: Sequence[str],
        outputs: str,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ) -> None:
        super().__init__(inputs, outputs)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, data: Sequence[mx.array], state: MutableMapping[str, Any]) -> mx.array:
        y_pred, y_true = data

        # Get probabilities via softmax if logits are provided
        probs = mx.softmax(y_pred, axis=-1)
        num_classes = y_pred.shape[-1]

        # Convert targets to one-hot if needed (vectorized)
        if y_true.ndim == 1 or y_true.shape[-1] != num_classes:
            # Ensure y_true is integer indices
            indices = y_true.flatten().astype(mx.int32)

            # Create one-hot encoding using scatter-like operation
            # eye matrix gives us one-hot rows that we can index into
            eye = mx.eye(num_classes)
            y_true_onehot = eye[indices]
        else:
            y_true_onehot = y_true.astype(mx.float32)

        # Compute cross entropy
        ce = -y_true_onehot * mx.log(mx.clip(probs, 1e-7, 1.0))

        # Get probability of true class
        p_t = mx.sum(probs * y_true_onehot, axis=-1, keepdims=True)

        # Compute focal weight
        focal_weight = mx.power(1.0 - p_t, self.gamma)

        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_weight = self.alpha * y_true_onehot + (1.0 - self.alpha) * (1.0 - y_true_onehot)
            focal_weight = focal_weight * alpha_weight

        # Compute focal loss
        focal_loss = focal_weight * ce
        focal_loss = mx.sum(focal_loss, axis=-1)

        if self.reduction == "mean":
            return mx.mean(focal_loss)
        elif self.reduction == "sum":
            return mx.sum(focal_loss)
        else:
            return focal_loss
