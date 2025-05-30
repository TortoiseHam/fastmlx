from __future__ import annotations

from typing import Tuple

import mlx.nn as nn
import mlx.core as mx


class LeNet(nn.Module):
    """Classic LeNet-5 style CNN for MNIST."""

    def __init__(self, input_shape: Tuple[int, int, int] = (1, 28, 28), classes: int = 10) -> None:
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool = nn.pooling.MaxPool2d((2, 2))
        flat_h = ((h - 2) // 2 - 2) // 2 - 2
        flat_w = ((w - 2) // 2 - 2) // 2 - 2
        self.fc1 = nn.Linear(flat_h * flat_w * 64, 64)
        self.fc2 = nn.Linear(64, classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x
