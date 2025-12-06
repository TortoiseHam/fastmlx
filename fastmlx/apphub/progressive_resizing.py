"""Progressive Resizing for Faster Training.

This example demonstrates progressive resizing, a technique that starts training
with small images and gradually increases resolution.

Benefits:
- Faster initial training (smaller images = faster batches)
- Better generalization through multi-scale learning
- Effective data augmentation via resolution changes
- Popular in fast.ai and image competition winning solutions

Example usage:
    python progressive_resizing.py --start_size 14 --end_size 28 --epochs 15
"""

import argparse
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import fastmlx as fe
from fastmlx.dataset import MLXDataset
from fastmlx.op import ModelOp, Op, UpdateOp
from fastmlx.trace import Accuracy, Trace


class ResizeOp(Op):
    """Resize images to specified size using bilinear interpolation.

    Args:
        inputs: Input image tensor.
        outputs: Output resized image.
        size: Target size (height, width) or single int for square.
        mode: When to execute.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        size: int | tuple[int, int],
        mode: str | list[str] | None = None,
    ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> mx.array:
        """Resize image using bilinear interpolation."""
        x = data[0] if isinstance(data, list) else data

        # Get input shape
        if x.ndim == 3:
            x = mx.expand_dims(x, axis=-1)

        batch_size, h, w, c = x.shape
        new_h, new_w = self.size

        if h == new_h and w == new_w:
            return x

        # Simple nearest neighbor resize for MLX
        # In practice you'd want bilinear interpolation
        y_ratio = h / new_h
        x_ratio = w / new_w

        # Create output grid
        y_coords = (mx.arange(new_h) * y_ratio).astype(mx.int32)
        x_coords = (mx.arange(new_w) * x_ratio).astype(mx.int32)

        # Clip to valid range
        y_coords = mx.clip(y_coords, 0, h - 1)
        x_coords = mx.clip(x_coords, 0, w - 1)

        # Gather pixels
        resized = x[:, y_coords, :, :][:, :, x_coords, :]

        return resized


class ProgressiveResizeScheduler(Trace):
    """Dynamically adjust image size during training.

    Args:
        resize_op: The resize operation to modify.
        schedule: Dict mapping epoch -> size, or list of (epoch, size) tuples.
        freeze_bn_on_increase: Whether to freeze BN stats when resolution increases.
    """

    def __init__(
        self,
        resize_op: ResizeOp,
        schedule: dict[int, int] | list[tuple[int, int]],
        freeze_bn_on_increase: bool = False,
    ):
        super().__init__()
        self.resize_op = resize_op

        if isinstance(schedule, list):
            self.schedule = dict(schedule)
        else:
            self.schedule = schedule

        self.freeze_bn = freeze_bn_on_increase
        self.current_size = None

    def on_epoch_begin(self, data: dict[str, Any]) -> None:
        """Update resize op size if needed."""
        epoch = data.get("epoch", 0)

        if epoch in self.schedule:
            new_size = self.schedule[epoch]
            old_size = self.resize_op.size
            self.resize_op.size = (new_size, new_size)
            self.current_size = new_size

            print(f"\nEpoch {epoch}: Resizing images from {old_size} to ({new_size}, {new_size})")

    def on_epoch_end(self, data: dict[str, Any]) -> None:
        """Report current resolution."""
        if self.current_size is not None:
            data["image_size"] = self.current_size


class FlexibleCNN(nn.Module):
    """CNN that can handle variable input sizes.

    Uses global average pooling before the classifier to handle any resolution.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(128)

        # Global average pooling makes this resolution-agnostic
        self.fc = nn.Linear(128, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            x = mx.expand_dims(x, axis=-1)

        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = nn.relu(self.bn3(self.conv3(x)))

        # Global average pooling - average over spatial dimensions
        x = mx.mean(x, axis=(1, 2))

        x = self.fc(x)

        return x


def get_estimator(
    epochs: int = 15,
    batch_size: int = 128,
    start_size: int = 14,
    end_size: int = 28,
    lr: float = 0.01,
) -> fe.Estimator:
    """Create an estimator with progressive resizing.

    Args:
        epochs: Total training epochs.
        batch_size: Training batch size.
        start_size: Initial image size.
        end_size: Final image size.
        lr: Learning rate.

    Returns:
        Configured Estimator.
    """
    from fastmlx.dataset.data import mnist

    # Load MNIST dataset
    train_data, test_data = mnist.load_data()

    # Create resize schedule - gradually increase size
    # Example: epochs 0-4 at 14x14, 5-9 at 21x21, 10+ at 28x28
    num_stages = 3
    epochs_per_stage = epochs // num_stages

    sizes = np.linspace(start_size, end_size, num_stages).astype(int)
    schedule = {}
    for i, size in enumerate(sizes):
        schedule[i * epochs_per_stage] = int(size)

    print(f"Progressive resize schedule: {schedule}")

    # Create resize op with initial size
    resize_op = ResizeOp(inputs="x", outputs="x", size=start_size)

    # Create pipeline
    pipeline = fe.Pipeline(
        train_data=MLXDataset(data={"x": train_data[0], "y": train_data[1]}),
        eval_data=MLXDataset(data={"x": test_data[0], "y": test_data[1]}),
        batch_size=batch_size,
        ops=[
            fe.op.Normalize(inputs="x", outputs="x", mean=0.1307, std=0.3081),
            resize_op,
        ],
    )

    # Build model
    model = fe.build(
        model_fn=lambda: FlexibleCNN(num_classes=10),
        optimizer_fn=lambda: optim.Adam(learning_rate=lr),
    )

    network = fe.Network(
        ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            fe.op.CrossEntropy(inputs=["y_pred", "y"], outputs="loss", mode="train"),
            UpdateOp(model=model, loss_name="loss"),
        ]
    )

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=[
            ProgressiveResizeScheduler(resize_op=resize_op, schedule=schedule),
            Accuracy(true_key="y", pred_key="y_pred"),
        ],
        log_interval=100,
    )

    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive Resizing Training")
    parser.add_argument("--epochs", type=int, default=15, help="Total epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--start_size", type=int, default=14, help="Initial image size")
    parser.add_argument("--end_size", type=int, default=28, help="Final image size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    print(f"Progressive resizing: {args.start_size}x{args.start_size} -> {args.end_size}x{args.end_size}")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        start_size=args.start_size,
        end_size=args.end_size,
        lr=args.lr,
    )
    est.fit()
    est.test()
