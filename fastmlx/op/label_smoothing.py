"""Label Smoothing loss operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .op import LossOp


class LabelSmoothingCrossEntropy(LossOp):
    """Cross entropy loss with label smoothing.

    Label smoothing helps prevent overconfidence by softening the target
    distribution. Instead of a one-hot target, it uses a mixture of the
    one-hot target and a uniform distribution.

    Args:
        inputs: Tuple of (predictions, targets) keys.
        outputs: Output key for the loss value.
        smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform).
        reduction: How to reduce the loss. Options: 'mean', 'sum', 'none'.

    Example:
        >>> loss_op = LabelSmoothingCrossEntropy(
        ...     inputs=("y_pred", "y"),
        ...     outputs="loss",
        ...     smoothing=0.1
        ... )
    """

    def __init__(
        self,
        inputs: Tuple[str, str],
        outputs: str,
        smoothing: float = 0.1,
        reduction: str = "mean"
    ) -> None:
        super().__init__(inputs, outputs)
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        data: Tuple[mx.array, mx.array],
        state: MutableMapping[str, Any]
    ) -> mx.array:
        y_pred, y_true = data

        # Get number of classes
        num_classes = y_pred.shape[-1]

        # Compute log softmax
        log_probs = mx.log(mx.softmax(y_pred, axis=-1) + 1e-8)

        # One-hot encode targets if needed
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[-1] == 1):
            y_true_flat = y_true.flatten().astype(mx.int32)
            eye = mx.eye(num_classes)
            targets = eye[y_true_flat]
        else:
            targets = y_true

        # Apply label smoothing
        # smooth_targets = (1 - smoothing) * one_hot + smoothing / num_classes
        smooth_targets = (1 - self.smoothing) * targets + self.smoothing / num_classes

        # Compute loss
        loss = -mx.sum(smooth_targets * log_probs, axis=-1)

        # Reduction
        if self.reduction == "mean":
            return mx.mean(loss)
        elif self.reduction == "sum":
            return mx.sum(loss)
        else:
            return loss


class WeightedCrossEntropy(LossOp):
    """Cross entropy loss with class weights for imbalanced datasets.

    Args:
        inputs: Tuple of (predictions, targets) keys.
        outputs: Output key for the loss value.
        weights: Class weights as a list or array. If None, uses uniform weights.
        reduction: How to reduce the loss. Options: 'mean', 'sum', 'none'.

    Example:
        >>> # For imbalanced binary classification
        >>> loss_op = WeightedCrossEntropy(
        ...     inputs=("y_pred", "y"),
        ...     outputs="loss",
        ...     weights=[1.0, 10.0]  # 10x weight for class 1
        ... )
    """

    def __init__(
        self,
        inputs: Tuple[str, str],
        outputs: str,
        weights: Optional[Union[list, mx.array]] = None,
        reduction: str = "mean"
    ) -> None:
        super().__init__(inputs, outputs)
        self.weights = mx.array(weights) if weights is not None else None
        self.reduction = reduction

    def forward(
        self,
        data: Tuple[mx.array, mx.array],
        state: MutableMapping[str, Any]
    ) -> mx.array:
        y_pred, y_true = data

        # Compute standard cross entropy
        loss = nn.losses.cross_entropy(y_pred, y_true, reduction="none")

        # Apply class weights
        if self.weights is not None:
            if y_true.ndim == 1:
                sample_weights = self.weights[y_true.astype(mx.int32)]
            else:
                sample_weights = mx.sum(y_true * self.weights, axis=-1)
            loss = loss * sample_weights

        # Reduction
        if self.reduction == "mean":
            return mx.mean(loss)
        elif self.reduction == "sum":
            return mx.sum(loss)
        else:
            return loss
