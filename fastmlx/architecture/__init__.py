"""Reference neural network architectures for FastMLX."""

from .lenet import LeNet
from .resnet9 import ResNet9
from .unet import UNet, AttentionUNet
from .wideresnet import (
    WideResNet,
    WideResNet16_8,
    WideResNet28_10,
    WideResNet40_4,
)

__all__ = [
    # Classification
    "LeNet",
    "ResNet9",
    "WideResNet",
    "WideResNet16_8",
    "WideResNet28_10",
    "WideResNet40_4",
    # Segmentation
    "UNet",
    "AttentionUNet",
]
