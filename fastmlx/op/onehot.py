"""One-hot encoding operation."""

from __future__ import annotations

from typing import Any, MutableMapping

import mlx.core as mx
import numpy as np

from .op import Op


class Onehot(Op):
    """Convert integer labels to one-hot vectors.

    Args:
        inputs: Input key name for integer labels.
        outputs: Output key name for one-hot vectors.
        num_classes: Number of classes (length of one-hot vector).
        label_smoothing: Amount of label smoothing to apply (0.0 to 1.0).
            Redistributes probability mass from true class to all classes.

    Example:
        >>> op = Onehot("y", "y_onehot", num_classes=10)
        >>> labels = mx.array([0, 3, 9])
        >>> onehot = op.forward(labels, {})  # Shape: (3, 10)

    Raises:
        ValueError: If any label is outside [0, num_classes).
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        num_classes: int,
        label_smoothing: float = 0.0
    ) -> None:
        super().__init__(inputs, outputs)
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        y = np.array(data).astype(np.int64)

        # Validate labels are in range
        if y.size > 0:
            min_val = int(y.min())
            max_val = int(y.max())
            if min_val < 0:
                raise ValueError(
                    f"Onehot: Labels must be >= 0, but found minimum value {min_val}"
                )
            if max_val >= self.num_classes:
                raise ValueError(
                    f"Onehot: Labels must be < num_classes ({self.num_classes}), "
                    f"but found maximum value {max_val}"
                )

        # Create one-hot encoding using eye matrix indexing
        oh = np.eye(self.num_classes, dtype=np.float32)[y]

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / self.num_classes
            oh = np.where(
                oh == 1.0,
                1.0 - self.label_smoothing + smooth,
                smooth
            )

        return mx.array(oh)
