"""Image classification examples.

- mnist: MNIST digit classification with LeNet
- cifar10: CIFAR-10 classification with ResNet9
- fashion_mnist: Fashion-MNIST classification with LeNet
- vit_cifar10: CIFAR-10 with Vision Transformer
- wideresnet_cifar10: CIFAR-10 with WideResNet variants
"""

from . import (
    cifar10,
    fashion_mnist,
    mnist,
    vit_cifar10,
    wideresnet_cifar10,
)

__all__ = [
    "mnist",
    "cifar10",
    "fashion_mnist",
    "vit_cifar10",
    "wideresnet_cifar10",
]
