"""Label Smoothing Regularization for Classification.

This example demonstrates label smoothing, a regularization technique that prevents
the model from becoming overconfident by softening the target labels.

Instead of hard targets [0, 1, 0], label smoothing uses soft targets like [0.05, 0.9, 0.05].
This helps with:
- Better generalization
- Reduced overconfidence
- Improved calibration

Example usage:
    python label_smoothing.py --smoothing 0.1 --epochs 10
"""

import argparse
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import fastmlx as fe
from fastmlx.dataset import MLXDataset
from fastmlx.op import ModelOp, Op, UpdateOp
from fastmlx.trace import Accuracy


class LabelSmoothingCrossEntropy(Op):
    """Cross entropy loss with label smoothing.

    Args:
        inputs: Name of the prediction tensor.
        outputs: Name for the loss output.
        smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform).
        mode: When to execute ("train", "eval", "test", "infer", or None for all).
    """

    def __init__(
        self,
        inputs: str | list[str],
        outputs: str | list[str],
        smoothing: float = 0.1,
        mode: str | list[str] | None = None,
    ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.smoothing = smoothing

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> mx.array:
        """Compute label smoothing cross entropy loss."""
        y_pred, y_true = data

        num_classes = y_pred.shape[-1]

        # Convert to one-hot if needed
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[-1] == 1):
            y_true = y_true.flatten().astype(mx.int32)
            one_hot = mx.one_hot(y_true, num_classes)
        else:
            one_hot = y_true

        # Apply label smoothing
        smooth_labels = one_hot * (1.0 - self.smoothing) + self.smoothing / num_classes

        # Log softmax for numerical stability
        log_probs = y_pred - mx.logsumexp(y_pred, axis=-1, keepdims=True)

        # Cross entropy with smooth labels
        loss = -mx.sum(smooth_labels * log_probs, axis=-1)

        return mx.mean(loss)


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def __call__(self, x: mx.array) -> mx.array:
        # Ensure correct shape [B, H, W, C]
        if x.ndim == 3:
            x = mx.expand_dims(x, axis=-1)

        # Conv block 1
        x = nn.relu(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # Conv block 2
        x = nn.relu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # Conv block 3
        x = nn.relu(self.conv3(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # Flatten and dense
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(nn.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def get_estimator(
    epochs: int = 10,
    batch_size: int = 128,
    smoothing: float = 0.1,
    lr: float = 0.001,
    compare: bool = False,
) -> fe.Estimator:
    """Create an estimator for label smoothing training.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        smoothing: Label smoothing factor (0.0-1.0).
        lr: Learning rate.
        compare: If True, train both with and without smoothing for comparison.

    Returns:
        Configured Estimator.
    """
    from fastmlx.dataset.data import mnist

    # Load MNIST dataset
    train_data, test_data = mnist.load_data()

    # Create pipeline
    pipeline = fe.Pipeline(
        train_data=MLXDataset(data={"x": train_data[0], "y": train_data[1]}),
        eval_data=MLXDataset(data={"x": test_data[0], "y": test_data[1]}),
        batch_size=batch_size,
        ops=[
            fe.op.Normalize(inputs="x", outputs="x", mean=0.1307, std=0.3081),
        ],
    )

    # Build model with label smoothing
    model_smooth = fe.build(
        model_fn=lambda: SimpleCNN(num_classes=10),
        optimizer_fn=lambda: optim.Adam(learning_rate=lr),
    )

    network_ops = [
        ModelOp(model=model_smooth, inputs="x", outputs="y_pred_smooth"),
        LabelSmoothingCrossEntropy(
            inputs=["y_pred_smooth", "y"],
            outputs="loss_smooth",
            smoothing=smoothing,
            mode="train",
        ),
        UpdateOp(model=model_smooth, loss_name="loss_smooth"),
    ]

    traces = [
        Accuracy(true_key="y", pred_key="y_pred_smooth", output_name="acc_smooth"),
    ]

    # Optionally add comparison model without smoothing
    if compare:
        model_baseline = fe.build(
            model_fn=lambda: SimpleCNN(num_classes=10),
            optimizer_fn=lambda: optim.Adam(learning_rate=lr),
        )

        network_ops.extend([
            ModelOp(model=model_baseline, inputs="x", outputs="y_pred_base"),
            fe.op.CrossEntropy(
                inputs=["y_pred_base", "y"],
                outputs="loss_base",
                mode="train",
            ),
            UpdateOp(model=model_baseline, loss_name="loss_base"),
        ])

        traces.append(
            Accuracy(true_key="y", pred_key="y_pred_base", output_name="acc_baseline"),
        )

    network = fe.Network(ops=network_ops)

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces,
        log_interval=100,
    )

    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Smoothing Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Smoothing factor")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline")
    args = parser.parse_args()

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        smoothing=args.smoothing,
        lr=args.lr,
        compare=args.compare,
    )
    est.fit()
    est.test()
