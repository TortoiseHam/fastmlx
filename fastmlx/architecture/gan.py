"""GAN architectures for generative modeling."""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class DCGANGenerator(nn.Module):
    """DCGAN Generator network.

    Transforms random noise vectors into images using transposed convolutions.

    Args:
        latent_dim: Dimension of input noise vector.
        image_shape: Output image shape as (channels, height, width).
        base_filters: Number of filters in the last conv layer.

    Example:
        >>> gen = DCGANGenerator(latent_dim=100, image_shape=(1, 28, 28))
        >>> z = mx.random.normal((4, 100))
        >>> images = gen(z)  # (4, 28, 28, 1)

    Reference:
        Radford et al., "Unsupervised Representation Learning with Deep
        Convolutional Generative Adversarial Networks", ICLR 2016.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        base_filters: int = 64
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        channels, height, width = image_shape

        # Calculate initial spatial size (after all upsampling)
        # For 28x28: start from 7x7, upsample 2x twice
        # For 32x32: start from 4x4, upsample 2x three times
        # For 64x64: start from 4x4, upsample 2x four times
        if height == 28:
            self.init_size = 7
            self.num_upsample = 2
        elif height == 32:
            self.init_size = 4
            self.num_upsample = 3
        elif height == 64:
            self.init_size = 4
            self.num_upsample = 4
        else:
            # Generic case
            self.init_size = 4
            self.num_upsample = 0
            size = 4
            while size < height:
                size *= 2
                self.num_upsample += 1

        # Initial dense layer
        init_channels = base_filters * (2 ** (self.num_upsample - 1))
        self.fc = nn.Linear(latent_dim, init_channels * self.init_size * self.init_size)
        self.init_channels = init_channels

        # Build upsampling blocks
        self.blocks = []
        in_ch = init_channels
        for i in range(self.num_upsample - 1):
            out_ch = in_ch // 2
            self.blocks.append(self._make_block(in_ch, out_ch))
            in_ch = out_ch

        # Final layer to image channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_ch, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create an upsampling block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm(out_channels),
            nn.ReLU(),
        )

    def __call__(self, z: mx.array) -> mx.array:
        """Generate images from noise.

        Args:
            z: Noise vector of shape (batch, latent_dim).

        Returns:
            Generated images of shape (batch, height, width, channels).
        """
        x = self.fc(z)
        x = x.reshape(-1, self.init_size, self.init_size, self.init_channels)
        x = nn.relu(x)

        for block in self.blocks:
            x = block(x)

        x = self.final(x)

        # Crop to exact size if needed
        _, h, w, _ = x.shape
        target_h, target_w = self.image_shape[1], self.image_shape[2]
        if h > target_h or w > target_w:
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            x = x[:, start_h:start_h + target_h, start_w:start_w + target_w, :]

        return x


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator network.

    Binary classifier that distinguishes real images from generated ones.

    Args:
        image_shape: Input image shape as (channels, height, width).
        base_filters: Number of filters in the first conv layer.

    Example:
        >>> disc = DCGANDiscriminator(image_shape=(1, 28, 28))
        >>> images = mx.random.normal((4, 28, 28, 1))
        >>> scores = disc(images)  # (4, 1)
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        base_filters: int = 64
    ) -> None:
        super().__init__()
        channels = image_shape[0]

        self.blocks = nn.Sequential(
            # First block (no batch norm)
            nn.Conv2d(channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            # Second block
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm(base_filters * 2),
            nn.LeakyReLU(negative_slope=0.2),
            # Third block
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm(base_filters * 4),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Calculate flattened size
        # For 28x28: 28->14->7->3 = 3x3x256
        # For 32x32: 32->16->8->4 = 4x4x256
        h, w = image_shape[1], image_shape[2]
        final_h = h // 8
        final_w = w // 8
        flat_size = base_filters * 4 * max(1, final_h) * max(1, final_w)

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Classify images as real or fake.

        Args:
            x: Images of shape (batch, height, width, channels).

        Returns:
            Logits of shape (batch, 1). Higher = more likely real.
        """
        features = self.blocks(x)
        features = features.reshape(features.shape[0], -1)
        return self.fc(features)


class SimpleGenerator(nn.Module):
    """Simple MLP-based generator for small images like MNIST.

    Faster to train than DCGAN for simple datasets.

    Args:
        latent_dim: Dimension of input noise vector.
        image_shape: Output image shape as (channels, height, width).
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        output_dim = image_shape[0] * image_shape[1] * image_shape[2]

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh(),
        )

    def __call__(self, z: mx.array) -> mx.array:
        """Generate images from noise."""
        x = self.net(z)
        return x.reshape(-1, self.image_shape[1], self.image_shape[2], self.image_shape[0])


class SimpleDiscriminator(nn.Module):
    """Simple MLP-based discriminator for small images.

    Args:
        image_shape: Input image shape as (channels, height, width).
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        hidden_dim: int = 256
    ) -> None:
        super().__init__()
        input_dim = image_shape[0] * image_shape[1] * image_shape[2]

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Classify images as real or fake."""
        x = x.reshape(x.shape[0], -1)
        return self.net(x)
