from __future__ import annotations

from typing import Any, List, MutableMapping, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from .op import LossOp


class CrossEntropy(LossOp):
    """Compute the mean cross entropy loss.

    This op computes softmax cross-entropy loss between predictions and labels.
    Handles both integer labels and one-hot encoded labels.

    Args:
        inputs: Tuple of (predictions_key, labels_key).
        outputs: Key to store the loss value.
        mode: When to execute (default: all modes).

    Example:
        >>> CrossEntropy(inputs=("y_pred", "y"), outputs="ce")
    """

    def __init__(
        self,
        inputs: Sequence[str],
        outputs: str,
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs, outputs, mode)

    def forward(self, data: Sequence[mx.array], state: MutableMapping[str, Any]) -> mx.array:
        y_pred, y_true = data

        # Handle one-hot encoded labels by converting to integer indices
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true = mx.argmax(y_true, axis=-1)

        loss = nn.losses.cross_entropy(y_pred, y_true)
        return mx.mean(loss)
