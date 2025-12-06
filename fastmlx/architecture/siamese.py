"""Siamese network architecture for one-shot learning."""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class SiameseNetwork(nn.Module):
    """Siamese Network for learning similarity between inputs.

    Uses a shared encoder network to map inputs to an embedding space,
    then computes similarity between embeddings. Suitable for one-shot
    learning, verification, and similarity tasks.

    Args:
        input_shape: Input shape as (channels, height, width).
        embedding_dim: Dimension of the embedding space.

    Example:
        >>> model = SiameseNetwork(input_shape=(1, 28, 28), embedding_dim=128)
        >>> x1 = mx.random.normal((4, 28, 28, 1))
        >>> x2 = mx.random.normal((4, 28, 28, 1))
        >>> emb1, emb2 = model(x1, x2)  # Both (4, 128)

    Reference:
        Koch et al., "Siamese Neural Networks for One-shot Image Recognition", 2015.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        embedding_dim: int = 128
    ) -> None:
        super().__init__()
        in_channels = input_shape[0]

        # Shared convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate flattened size (assuming 28x28 input -> 3x3 after pooling)
        # For 28x28: after 3 pooling layers -> 3x3
        h, w = input_shape[1] // 8, input_shape[2] // 8
        flat_size = 256 * max(1, h) * max(1, w)

        # Embedding layer
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

    def encode(self, x: mx.array) -> mx.array:
        """Encode a single input to embedding space.

        Args:
            x: Input of shape (batch, height, width, channels).

        Returns:
            Embedding of shape (batch, embedding_dim).
        """
        features = self.encoder(x)
        # Flatten
        features = features.reshape(features.shape[0], -1)
        embedding = self.fc(features)
        return embedding

    def __call__(
        self,
        x1: mx.array,
        x2: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass for a pair of inputs.

        Args:
            x1: First input of shape (batch, height, width, channels).
            x2: Second input of shape (batch, height, width, channels).

        Returns:
            Tuple of embeddings (emb1, emb2), each of shape (batch, embedding_dim).
        """
        emb1 = self.encode(x1)
        emb2 = self.encode(x2)
        return emb1, emb2


class SiameseEncoder(nn.Module):
    """Standalone encoder for Siamese network inference.

    This is the same encoder used in SiameseNetwork but designed for
    single-input inference after training.

    Args:
        input_shape: Input shape as (channels, height, width).
        embedding_dim: Dimension of the embedding space.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        embedding_dim: int = 128
    ) -> None:
        super().__init__()
        in_channels = input_shape[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        h, w = input_shape[1] // 8, input_shape[2] // 8
        flat_size = 256 * max(1, h) * max(1, w)

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Encode input to embedding space.

        Args:
            x: Input of shape (batch, height, width, channels).

        Returns:
            Embedding of shape (batch, embedding_dim).
        """
        features = self.encoder(x)
        features = features.reshape(features.shape[0], -1)
        return self.fc(features)
