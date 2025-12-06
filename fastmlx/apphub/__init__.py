"""Example applications built with fastmlx.

Image Classification:
    - mnist: MNIST digit classification with LeNet
    - cifar10: CIFAR-10 classification with ResNet9
    - fashion_mnist: Fashion-MNIST classification with LeNet
    - vit_cifar10: CIFAR-10 with Vision Transformer
    - wideresnet_cifar10: CIFAR-10 with WideResNet variants

Image Segmentation:
    - unet_segmentation: UNet on synthetic circular mask data

Training Techniques:
    - super_convergence: 1cycle learning rate policy for fast training
    - fgsm_adversarial: FGSM adversarial training for robustness
    - mixup_training: MixUp data augmentation
    - early_stopping: Adaptive training with early stopping
    - lr_finder: Learning rate range test
    - multitask_learning: Uncertainty-weighted multi-task learning

Metric Learning:
    - siamese_mnist: Siamese network for one-shot learning

Self-Supervised Learning:
    - simclr_cifar10: SimCLR contrastive learning

Generative Models:
    - autoencoder_mnist: Autoencoder and VAE on MNIST
    - dcgan_mnist: Deep Convolutional GAN

Language Modeling:
    - gpt_language_model: Character-level GPT on Shakespeare

Tabular Data:
    - tabular_dnn: DNN for structured data (classification/regression)
"""

from . import (
    mnist,
    cifar10,
    fashion_mnist,
    vit_cifar10,
    wideresnet_cifar10,
    unet_segmentation,
    super_convergence,
    fgsm_adversarial,
    mixup_training,
    early_stopping,
    lr_finder,
    multitask_learning,
    siamese_mnist,
    simclr_cifar10,
    autoencoder_mnist,
    dcgan_mnist,
    gpt_language_model,
    tabular_dnn,
)

__all__ = [
    # Image Classification
    "mnist",
    "cifar10",
    "fashion_mnist",
    "vit_cifar10",
    "wideresnet_cifar10",
    # Image Segmentation
    "unet_segmentation",
    # Training Techniques
    "super_convergence",
    "fgsm_adversarial",
    "mixup_training",
    "early_stopping",
    "lr_finder",
    "multitask_learning",
    # Metric Learning
    "siamese_mnist",
    # Self-Supervised Learning
    "simclr_cifar10",
    # Generative Models
    "autoencoder_mnist",
    "dcgan_mnist",
    # Language Modeling
    "gpt_language_model",
    # Tabular Data
    "tabular_dnn",
]
