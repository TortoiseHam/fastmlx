"""Autoencoder architectures for unsupervised learning."""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class Autoencoder(nn.Module):
    """Convolutional Autoencoder for image reconstruction.

    Learns a compressed latent representation of images through
    an encoder-decoder architecture with reconstruction loss.

    Args:
        input_shape: Input shape as (channels, height, width).
        latent_dim: Dimension of the latent space.

    Example:
        >>> model = Autoencoder(input_shape=(1, 28, 28), latent_dim=32)
        >>> x = mx.random.normal((4, 28, 28, 1))
        >>> x_recon = model(x)  # (4, 28, 28, 1)
        >>> z = model.encode(x)  # (4, 32)
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        latent_dim: int = 32
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        in_channels = input_shape[0]

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate feature map size after convolutions
        # For 28x28: 28->14->7->4 (with padding)
        self._feat_h = (input_shape[1] + 7) // 8
        self._feat_w = (input_shape[2] + 7) // 8
        self._feat_size = 128 * self._feat_h * self._feat_w

        self.encoder_fc = nn.Linear(self._feat_size, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self._feat_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: mx.array) -> mx.array:
        """Encode input to latent space.

        Args:
            x: Input of shape (batch, height, width, channels).

        Returns:
            Latent representation of shape (batch, latent_dim).
        """
        h = self.encoder_conv(x)
        h = h.reshape(h.shape[0], -1)
        z = self.encoder_fc(h)
        return z

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent representation to image.

        Args:
            z: Latent representation of shape (batch, latent_dim).

        Returns:
            Reconstructed image of shape (batch, height, width, channels).
        """
        h = self.decoder_fc(z)
        h = h.reshape(-1, self._feat_h, self._feat_w, 128)
        x_recon = self.decoder_conv(h)
        # Crop or pad to exact input size
        x_recon = x_recon[:, :self.input_shape[1], :self.input_shape[2], :]
        return x_recon

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: encode then decode.

        Args:
            x: Input of shape (batch, height, width, channels).

        Returns:
            Reconstructed input of shape (batch, height, width, channels).
        """
        z = self.encode(x)
        return self.decode(z)


class VAE(nn.Module):
    """Variational Autoencoder for generative modeling.

    Learns a probabilistic latent representation using the
    reparameterization trick. Can generate new samples by
    sampling from the learned latent distribution.

    Args:
        input_shape: Input shape as (channels, height, width).
        latent_dim: Dimension of the latent space.

    Example:
        >>> model = VAE(input_shape=(1, 28, 28), latent_dim=32)
        >>> x = mx.random.normal((4, 28, 28, 1))
        >>> x_recon, mu, log_var = model(x)
        >>> z = model.sample(16)  # Sample 16 random latent vectors
        >>> generated = model.decode(z)  # Generate images

    Reference:
        Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        latent_dim: int = 32
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        in_channels = input_shape[0]

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self._feat_h = (input_shape[1] + 7) // 8
        self._feat_w = (input_shape[2] + 7) // 8
        self._feat_size = 128 * self._feat_h * self._feat_w

        # Latent space parameters (mean and log variance)
        self.fc_mu = nn.Linear(self._feat_size, latent_dim)
        self.fc_log_var = nn.Linear(self._feat_size, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self._feat_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input of shape (batch, height, width, channels).

        Returns:
            Tuple of (mu, log_var), each of shape (batch, latent_dim).
        """
        h = self.encoder_conv(x)
        h = h.reshape(h.shape[0], -1)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: mx.array, log_var: mx.array) -> mx.array:
        """Sample from latent distribution using reparameterization trick.

        Args:
            mu: Mean of shape (batch, latent_dim).
            log_var: Log variance of shape (batch, latent_dim).

        Returns:
            Sampled latent vector of shape (batch, latent_dim).
        """
        std = mx.exp(0.5 * log_var)
        eps = mx.random.normal(mu.shape)
        return mu + eps * std

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent representation to image.

        Args:
            z: Latent representation of shape (batch, latent_dim).

        Returns:
            Reconstructed image of shape (batch, height, width, channels).
        """
        h = self.decoder_fc(z)
        h = h.reshape(-1, self._feat_h, self._feat_w, 128)
        x_recon = self.decoder_conv(h)
        x_recon = x_recon[:, :self.input_shape[1], :self.input_shape[2], :]
        return x_recon

    def sample(self, num_samples: int) -> mx.array:
        """Sample random latent vectors from standard normal.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Latent vectors of shape (num_samples, latent_dim).
        """
        return mx.random.normal((num_samples, self.latent_dim))

    def __call__(
        self,
        x: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass with reparameterization.

        Args:
            x: Input of shape (batch, height, width, channels).

        Returns:
            Tuple of (x_recon, mu, log_var).
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
