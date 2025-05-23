from typing import Tuple
import numpy as np
from mlx.data.datasets import mnist as mlx_mnist
from ..numpy_dataset import NumpyDataset


def load_data(image_key: str = "x", label_key: str = "y") -> Tuple[NumpyDataset, NumpyDataset]:
    train_buffer = mlx_mnist.load_mnist(train=True)
    test_buffer = mlx_mnist.load_mnist(train=False)
    train_images = np.stack([d["image"] for d in train_buffer])
    train_labels = np.array([d["label"] for d in train_buffer])
    test_images = np.stack([d["image"] for d in test_buffer])
    test_labels = np.array([d["label"] for d in test_buffer])
    train = NumpyDataset({image_key: train_images, label_key: train_labels})
    test = NumpyDataset({image_key: test_images, label_key: test_labels})
    return train, test
