"""UNet architecture for image segmentation."""

from __future__ import annotations

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block used in UNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock followed by MaxPool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        features = self.conv_block(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """Decoder block: Upsample, concatenate skip connection, ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Use transposed convolution for upsampling
        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv_block = ConvBlock(out_channels * 2, out_channels)

    def __call__(self, x: mx.array, skip: mx.array) -> mx.array:
        x = self.up_conv(x)
        # Handle size mismatches by cropping skip connection
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            diff_h = skip.shape[1] - x.shape[1]
            diff_w = skip.shape[2] - x.shape[2]
            skip = skip[:, diff_h // 2:skip.shape[1] - (diff_h - diff_h // 2),
                       diff_w // 2:skip.shape[2] - (diff_w - diff_w // 2), :]
        x = mx.concatenate([x, skip], axis=-1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """UNet architecture for image segmentation.

    Args:
        input_shape: Input image shape as (channels, height, width).
        classes: Number of output segmentation classes.
        base_filters: Number of filters in the first layer (doubles each encoder stage).
        depth: Number of encoder/decoder stages.

    Reference:
        Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
        Image Segmentation", MICCAI 2015.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 256, 256),
        classes: int = 2,
        base_filters: int = 64,
        depth: int = 4
    ) -> None:
        super().__init__()
        self.depth = depth
        in_channels = input_shape[0]

        # Encoder path
        self.encoders = []
        current_channels = in_channels
        for i in range(depth):
            out_channels = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(current_channels, out_channels))
            current_channels = out_channels

        # Bottleneck
        bottleneck_channels = base_filters * (2 ** depth)
        self.bottleneck = ConvBlock(current_channels, bottleneck_channels)

        # Decoder path
        self.decoders = []
        current_channels = bottleneck_channels
        for i in range(depth - 1, -1, -1):
            out_channels = base_filters * (2 ** i)
            self.decoders.append(DecoderBlock(current_channels, out_channels))
            current_channels = out_channels

        # Output layer
        self.output_conv = nn.Conv2d(base_filters, classes, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        # Encoder path with skip connections
        skip_connections: List[mx.array] = []
        for encoder in self.encoders:
            x, features = encoder(x)
            skip_connections.append(features)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])

        # Output
        x = self.output_conv(x)
        return x


class AttentionBlock(nn.Module):
    """Attention gate for AttentionUNet."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.W_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm(inter_channels)

    def __call__(self, gate: mx.array, skip: mx.array) -> mx.array:
        g = self.W_g(gate)
        x = self.W_x(skip)

        # Handle size mismatches
        if g.shape[1] != x.shape[1] or g.shape[2] != x.shape[2]:
            # Upsample gate to match skip
            g = mx.repeat(mx.repeat(g, 2, axis=1), 2, axis=2)
            if g.shape[1] > x.shape[1]:
                g = g[:, :x.shape[1], :, :]
            if g.shape[2] > x.shape[2]:
                g = g[:, :, :x.shape[2], :]

        psi = nn.relu(self.bn(g + x))
        psi = mx.sigmoid(self.psi(psi))
        return skip * psi


class AttentionDecoderBlock(nn.Module):
    """Decoder block with attention for AttentionUNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.attention = AttentionBlock(out_channels, out_channels, out_channels // 2)
        self.conv_block = ConvBlock(out_channels * 2, out_channels)

    def __call__(self, x: mx.array, skip: mx.array) -> mx.array:
        x = self.up_conv(x)
        # Apply attention
        skip = self.attention(x, skip)
        # Handle size mismatches
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            diff_h = skip.shape[1] - x.shape[1]
            diff_w = skip.shape[2] - x.shape[2]
            skip = skip[:, diff_h // 2:skip.shape[1] - (diff_h - diff_h // 2),
                       diff_w // 2:skip.shape[2] - (diff_w - diff_w // 2), :]
        x = mx.concatenate([x, skip], axis=-1)
        x = self.conv_block(x)
        return x


class AttentionUNet(nn.Module):
    """UNet with attention gates for improved segmentation.

    Args:
        input_shape: Input image shape as (channels, height, width).
        classes: Number of output segmentation classes.
        base_filters: Number of filters in the first layer.
        depth: Number of encoder/decoder stages.

    Reference:
        Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", 2018.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 256, 256),
        classes: int = 2,
        base_filters: int = 64,
        depth: int = 4
    ) -> None:
        super().__init__()
        self.depth = depth
        in_channels = input_shape[0]

        # Encoder path
        self.encoders = []
        current_channels = in_channels
        for i in range(depth):
            out_channels = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(current_channels, out_channels))
            current_channels = out_channels

        # Bottleneck
        bottleneck_channels = base_filters * (2 ** depth)
        self.bottleneck = ConvBlock(current_channels, bottleneck_channels)

        # Decoder path with attention
        self.decoders = []
        current_channels = bottleneck_channels
        for i in range(depth - 1, -1, -1):
            out_channels = base_filters * (2 ** i)
            self.decoders.append(AttentionDecoderBlock(current_channels, out_channels))
            current_channels = out_channels

        # Output layer
        self.output_conv = nn.Conv2d(base_filters, classes, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        # Encoder path with skip connections
        skip_connections: List[mx.array] = []
        for encoder in self.encoders:
            x, features = encoder(x)
            skip_connections.append(features)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with attention and skip connections
        skip_connections = skip_connections[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])

        # Output
        x = self.output_conv(x)
        return x
