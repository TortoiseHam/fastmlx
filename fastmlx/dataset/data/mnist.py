"""Utilities for loading the MNIST dataset using MLX."""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
from mlx.data.datasets import mnist as mlx_mnist
from ..mlx_dataset import MLXDataset


def load_data(image_key: str = "x", label_key: str = "y") -> Tuple[MLXDataset, MLXDataset]:
    train_buffer = mlx_mnist.load_mnist(train=True)
    test_buffer = mlx_mnist.load_mnist(train=False)
    train_images = mx.stack([mx.array(d["image"]) for d in train_buffer])
    train_labels = mx.array([int(d["label"]) for d in train_buffer])
    test_images = mx.stack([mx.array(d["image"]) for d in test_buffer])
    test_labels = mx.array([int(d["label"]) for d in test_buffer])
    train = MLXDataset({image_key: train_images, label_key: train_labels})
    test = MLXDataset({image_key: test_images, label_key: test_labels})
    return train, test
