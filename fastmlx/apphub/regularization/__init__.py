"""Regularization and training technique examples.

- mixup_training: MixUp data augmentation
- cutmix_training: CutMix data augmentation
- early_stopping: Adaptive training with early stopping
- focal_loss_imbalanced: Focal loss for class imbalance
- label_smoothing: Label smoothing regularization
- gradient_clipping: Gradient clipping for stable training
- gradient_accumulation: Training with large effective batch sizes
- pseudo_labeling: Semi-supervised learning with pseudo labels
- stochastic_weight_averaging: SWA for better generalization
- progressive_resizing: Multi-scale training with increasing resolution
"""

from . import (
    cutmix_training,
    early_stopping,
    focal_loss_imbalanced,
    gradient_accumulation,
    gradient_clipping,
    label_smoothing,
    mixup_training,
    progressive_resizing,
    pseudo_labeling,
    stochastic_weight_averaging,
)

__all__ = [
    "mixup_training",
    "cutmix_training",
    "early_stopping",
    "focal_loss_imbalanced",
    "label_smoothing",
    "gradient_clipping",
    "gradient_accumulation",
    "pseudo_labeling",
    "stochastic_weight_averaging",
    "progressive_resizing",
]
