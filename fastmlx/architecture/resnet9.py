from __future__ import annotations

from typing import Tuple

import mlx.nn as nn
import mlx.core as mx


class Residual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm(channels, momentum=0.2)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm(channels, momentum=0.2)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = nn.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        return x


class ResNet9(nn.Module):
    """A small ResNet model for CIFAR10."""

    def __init__(self, input_shape: Tuple[int, int, int] = (3, 32, 32), classes: int = 10) -> None:
        super().__init__()
        c, h, w = input_shape
        self.conv0 = nn.Conv2d(c, 64, 3, padding=1)
        self.bn0 = nn.BatchNorm(64, momentum=0.2)
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm(128, momentum=0.2)
        self.res1 = Residual(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm(256, momentum=0.2)
        self.res2 = Residual(256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm(512, momentum=0.2)
        self.res3 = Residual(512)
        self.pool = nn.pooling.MaxPool2d((2, 2))
        self.fc = nn.Linear(512, classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.leaky_relu(self.bn0(self.conv0(x)), negative_slope=0.1)
        x = self.pool(x)
        x = nn.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = self.pool(x)
        x = x + self.res1(x)
        x = nn.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = self.pool(x)
        x = x + self.res2(x)
        x = nn.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        x = self.pool(x)
        x = x + self.res3(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.softmax(self.fc(x), axis=-1)
        return x
