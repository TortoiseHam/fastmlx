"""Utilities for loading the CIFAR-10 dataset using MLX."""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
from mlx.data.datasets import cifar

from ..mlx_dataset import MLXDataset


def load_data(image_key: str = "x", label_key: str = "y") -> Tuple[MLXDataset, MLXDataset]:
    train_buffer = cifar.load_cifar10(train=True)
    test_buffer = cifar.load_cifar10(train=False)
    train_images = mx.stack([mx.array(d["image"]) for d in train_buffer])
    # Cast labels to Python ints in case the underlying dataset stores them as
    # numpy scalar types.  This mirrors the handling in the MNIST loader and
    # avoids ``mx.array`` complaining about unsupported ``ndarray`` inputs.
    train_labels = mx.array([int(d["label"]) for d in train_buffer])
    test_images = mx.stack([mx.array(d["image"]) for d in test_buffer])
    test_labels = mx.array([int(d["label"]) for d in test_buffer])
    train = MLXDataset({image_key: train_images, label_key: train_labels})
    test = MLXDataset({image_key: test_images, label_key: test_labels})
    return train, test
