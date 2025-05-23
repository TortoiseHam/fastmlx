"""Utilities for loading the CIFAR-10 dataset using MLX."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from mlx.data.datasets import cifar
from ..numpy_dataset import NumpyDataset


def load_data(image_key: str = "x", label_key: str = "y") -> Tuple[NumpyDataset, NumpyDataset]:
    train_buffer = cifar.load_cifar10(train=True)
    test_buffer = cifar.load_cifar10(train=False)
    train_images = np.stack([d["image"] for d in train_buffer])
    train_labels = np.array([d["label"] for d in train_buffer])
    test_images = np.stack([d["image"] for d in test_buffer])
    test_labels = np.array([d["label"] for d in test_buffer])
    train = NumpyDataset({image_key: train_images, label_key: train_labels})
    test = NumpyDataset({image_key: test_images, label_key: test_labels})
    return train, test
