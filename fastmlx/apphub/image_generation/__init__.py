"""Image generation examples.

- autoencoder_mnist: Autoencoder and VAE on MNIST
- dcgan_mnist: Deep Convolutional GAN
- conditional_gan: Class-conditional GAN
"""

from . import (
    autoencoder_mnist,
    conditional_gan,
    dcgan_mnist,
)

__all__ = [
    "autoencoder_mnist",
    "dcgan_mnist",
    "conditional_gan",
]
