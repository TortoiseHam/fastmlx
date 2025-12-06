"""Utilities for loading the Fashion-MNIST dataset using MLX."""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import mlx.core as mx

from ..mlx_dataset import MLXDataset


_BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_file(url: str, destination: str) -> None:
    """Download a file if it doesn't exist."""
    if os.path.exists(destination):
        return
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, destination)


def _read_idx_images(filepath: str) -> np.ndarray:
    """Read IDX image file format."""
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
    return images


def _read_idx_labels(filepath: str) -> np.ndarray:
    """Read IDX label file format."""
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_data(
    root_dir: str | None = None,
    image_key: str = "x",
    label_key: str = "y",
) -> Tuple[MLXDataset, MLXDataset]:
    """Load the Fashion-MNIST dataset.

    Fashion-MNIST is a dataset of Zalando's article images consisting of
    60,000 training examples and 10,000 test examples. Each example is a
    28x28 grayscale image associated with a label from 10 classes.

    Classes:
        0: T-shirt/top
        1: Trouser
        2: Pullover
        3: Dress
        4: Coat
        5: Sandal
        6: Shirt
        7: Sneaker
        8: Bag
        9: Ankle boot

    Args:
        root_dir: Directory to store the downloaded data.
                  Defaults to ``~/fastmlx_data/fashion_mnist``.
        image_key: Key name for images in the returned datasets.
        label_key: Key name for labels in the returned datasets.

    Returns:
        A tuple of training and test datasets.

    Reference:
        Xiao et al., "Fashion-MNIST: a Novel Image Dataset for Benchmarking
        Machine Learning Algorithms", 2017.
    """
    home = str(Path.home())
    if root_dir is None:
        root_dir = os.path.join(home, "fastmlx_data", "fashion_mnist")
    os.makedirs(root_dir, exist_ok=True)

    # Download files if needed
    for key, filename in _FILES.items():
        url = _BASE_URL + filename
        destination = os.path.join(root_dir, filename)
        _download_file(url, destination)

    # Load data
    train_images = _read_idx_images(os.path.join(root_dir, _FILES["train_images"]))
    train_labels = _read_idx_labels(os.path.join(root_dir, _FILES["train_labels"]))
    test_images = _read_idx_images(os.path.join(root_dir, _FILES["test_images"]))
    test_labels = _read_idx_labels(os.path.join(root_dir, _FILES["test_labels"]))

    train = MLXDataset({
        image_key: mx.array(train_images),
        label_key: mx.array(train_labels)
    })
    test = MLXDataset({
        image_key: mx.array(test_images),
        label_key: mx.array(test_labels)
    })

    return train, test
