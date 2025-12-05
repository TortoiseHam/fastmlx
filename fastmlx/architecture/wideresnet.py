"""Wide Residual Network (WideResNet) architecture."""

from __future__ import annotations

from typing import Tuple, Optional

import mlx.nn as nn
import mlx.core as mx


class BasicBlock(nn.Module):
    """Basic residual block for WideResNet.

    Uses two 3x3 convolutions with batch normalization and optional dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.0
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout_rate = dropout_rate

        # Shortcut connection
        self.shortcut: Optional[nn.Conv2d] = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        out = nn.relu(self.bn1(x))

        if self.shortcut is not None:
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)
        out = nn.relu(self.bn2(out))

        if self.dropout_rate > 0 and training:
            out = nn.Dropout(self.dropout_rate)(out)

        out = self.conv2(out)
        out = out + shortcut
        return out


class WideResNetGroup(nn.Module):
    """A group of residual blocks with the same number of channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        dropout_rate: float = 0.0
    ) -> None:
        super().__init__()
        self.blocks = []

        # First block may have stride and channel change
        self.blocks.append(BasicBlock(in_channels, out_channels, stride, dropout_rate))

        # Remaining blocks
        for _ in range(1, num_blocks):
            self.blocks.append(BasicBlock(out_channels, out_channels, 1, dropout_rate))

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        for block in self.blocks:
            x = block(x, training)
        return x


class WideResNet(nn.Module):
    """Wide Residual Network for image classification.

    Args:
        input_shape: Input image shape as (channels, height, width).
        classes: Number of output classes.
        depth: Total depth of the network. Must be 6n+4 for some integer n.
        widen_factor: Width multiplier for the residual blocks.
        dropout_rate: Dropout probability (0 to disable).

    Example depths:
        - WRN-16-8: depth=16, widen_factor=8
        - WRN-28-10: depth=28, widen_factor=10
        - WRN-40-4: depth=40, widen_factor=4

    Reference:
        Zagoruyko & Komodakis, "Wide Residual Networks", BMVC 2016.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        classes: int = 10,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.3
    ) -> None:
        super().__init__()

        # Validate depth
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4 for some integer n"
        n = (depth - 4) // 6  # Number of blocks per group

        # Channel widths
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        in_channels = input_shape[0]

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1)

        # Three groups of residual blocks
        self.group1 = WideResNetGroup(channels[0], channels[1], n, stride=1, dropout_rate=dropout_rate)
        self.group2 = WideResNetGroup(channels[1], channels[2], n, stride=2, dropout_rate=dropout_rate)
        self.group3 = WideResNetGroup(channels[2], channels[3], n, stride=2, dropout_rate=dropout_rate)

        # Final batch norm and classifier
        self.bn = nn.BatchNorm(channels[3])
        self.fc = nn.Linear(channels[3], classes)

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        # Initial conv
        out = self.conv1(x)

        # Residual groups
        out = self.group1(out, training)
        out = self.group2(out, training)
        out = self.group3(out, training)

        # Final layers
        out = nn.relu(self.bn(out))

        # Global average pooling
        out = mx.mean(out, axis=(1, 2))

        # Classifier
        out = self.fc(out)
        return out


class WideResNet16_8(WideResNet):
    """WRN-16-8: Depth 16, width factor 8."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        classes: int = 10,
        dropout_rate: float = 0.3
    ) -> None:
        super().__init__(input_shape, classes, depth=16, widen_factor=8, dropout_rate=dropout_rate)


class WideResNet28_10(WideResNet):
    """WRN-28-10: Depth 28, width factor 10. Common configuration for CIFAR."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        classes: int = 10,
        dropout_rate: float = 0.3
    ) -> None:
        super().__init__(input_shape, classes, depth=28, widen_factor=10, dropout_rate=dropout_rate)


class WideResNet40_4(WideResNet):
    """WRN-40-4: Depth 40, width factor 4."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        classes: int = 10,
        dropout_rate: float = 0.3
    ) -> None:
        super().__init__(input_shape, classes, depth=40, widen_factor=4, dropout_rate=dropout_rate)
