"""Example applications built with fastmlx.

Image Classification:
    - mnist: MNIST digit classification with LeNet
    - cifar10: CIFAR-10 classification with ResNet9
    - fashion_mnist: Fashion-MNIST classification with LeNet
    - vit_cifar10: CIFAR-10 with Vision Transformer

Image Segmentation:
    - unet_segmentation: UNet on synthetic circular mask data

Training Techniques:
    - super_convergence: 1cycle learning rate policy for fast training
    - fgsm_adversarial: FGSM adversarial training for robustness

Metric Learning:
    - siamese_mnist: Siamese network for one-shot learning

Generative Models:
    - autoencoder_mnist: Autoencoder and VAE on MNIST

Tabular Data:
    - tabular_dnn: DNN for structured data (classification/regression)
"""

from . import (
    mnist,
    cifar10,
    fashion_mnist,
    vit_cifar10,
    unet_segmentation,
    super_convergence,
    fgsm_adversarial,
    siamese_mnist,
    autoencoder_mnist,
    tabular_dnn,
)

__all__ = [
    # Image Classification
    "mnist",
    "cifar10",
    "fashion_mnist",
    "vit_cifar10",
    # Image Segmentation
    "unet_segmentation",
    # Training Techniques
    "super_convergence",
    "fgsm_adversarial",
    # Metric Learning
    "siamese_mnist",
    # Generative Models
    "autoencoder_mnist",
    # Tabular Data
    "tabular_dnn",
]
