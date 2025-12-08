"""Stochastic Weight Averaging (SWA) for Better Generalization.

This example demonstrates Stochastic Weight Averaging, a technique that improves
generalization by averaging model weights during training.

SWA works by:
1. Training normally for initial epochs
2. After a trigger epoch, start averaging weights
3. Average weights contribute to a smoother loss landscape
4. Use averaged weights for final model

Benefits:
- Better generalization
- Flatter minima
- Often improves accuracy by 1-2%

Reference: Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization"

Example usage:
    python stochastic_weight_averaging.py --swa_start 10 --epochs 20
"""

import argparse
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import fastmlx as fe
from fastmlx.dataset import MLXDataset
from fastmlx.op import ModelOp, UpdateOp
from fastmlx.trace import Accuracy, Trace


class SWATrace(Trace):
    """Stochastic Weight Averaging trace.

    Maintains a running average of model weights after a specified epoch.

    Args:
        model: The model to average.
        swa_start: Epoch to start averaging (0-indexed).
        swa_freq: How often to update average (in epochs).
        swa_lr: Optional constant LR during SWA phase.
    """

    def __init__(
        self,
        model: fe.build,
        swa_start: int = 10,
        swa_freq: int = 1,
        swa_lr: float | None = None,
    ):
        super().__init__()
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.swa_weights = None
        self.num_averaged = 0

    def on_epoch_begin(self, data: dict[str, Any]) -> None:
        """Optionally switch to SWA learning rate."""
        epoch = data.get("epoch", 0)

        if epoch >= self.swa_start and self.swa_lr is not None:
            # Set constant learning rate for SWA phase
            # Note: This is a simplified version - in practice you might
            # use a cyclic learning rate during SWA
            pass

    def on_epoch_end(self, data: dict[str, Any]) -> None:
        """Update SWA weights at end of epoch."""
        epoch = data.get("epoch", 0)

        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            self._update_swa_weights()
            data["swa_num_averaged"] = self.num_averaged

    def _update_swa_weights(self) -> None:
        """Update running average of weights."""
        current_weights = self.model.model.parameters()

        if self.swa_weights is None:
            # First averaging step - copy weights
            self.swa_weights = self._deep_copy_params(current_weights)
            self.num_averaged = 1
        else:
            # Update running average: swa = (swa * n + current) / (n + 1)
            self.num_averaged += 1
            self._update_average(self.swa_weights, current_weights, self.num_averaged)

    def _deep_copy_params(self, params: dict) -> dict:
        """Create a deep copy of parameters."""
        copied = {}
        for key, value in params.items():
            if isinstance(value, mx.array):
                copied[key] = mx.array(value)
            elif isinstance(value, dict):
                copied[key] = self._deep_copy_params(value)
            else:
                copied[key] = value
        return copied

    def _update_average(self, avg: dict, current: dict, n: int) -> None:
        """Update running average in place."""
        for key in avg.keys():
            if isinstance(avg[key], mx.array):
                # Running average: avg = avg + (current - avg) / n
                avg[key] = avg[key] + (current[key] - avg[key]) / n
            elif isinstance(avg[key], dict):
                self._update_average(avg[key], current[key], n)

    def on_end(self, data: dict[str, Any]) -> None:
        """Apply SWA weights to model at end of training."""
        if self.swa_weights is not None and self.num_averaged > 0:
            print(f"\nApplying SWA weights (averaged {self.num_averaged} checkpoints)")
            self._apply_weights(self.model.model, self.swa_weights)

    def _apply_weights(self, model: nn.Module, weights: dict) -> None:
        """Apply averaged weights to model."""
        model.update(weights)


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            x = mx.expand_dims(x, axis=-1)

        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = nn.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(nn.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def get_estimator(
    epochs: int = 20,
    batch_size: int = 128,
    swa_start: int = 10,
    swa_freq: int = 1,
    lr: float = 0.01,
    swa_lr: float | None = 0.001,
) -> fe.Estimator:
    """Create an estimator with Stochastic Weight Averaging.

    Args:
        epochs: Total training epochs.
        batch_size: Training batch size.
        swa_start: Epoch to start averaging.
        swa_freq: Averaging frequency (epochs).
        lr: Initial learning rate.
        swa_lr: Learning rate during SWA phase.

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

    # Build model with cosine annealing LR
    model = fe.build(
        model_fn=lambda: SimpleCNN(num_classes=10),
        optimizer_fn=lambda: optim.SGD(learning_rate=lr, momentum=0.9, weight_decay=1e-4),
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
            SWATrace(
                model=model,
                swa_start=swa_start,
                swa_freq=swa_freq,
                swa_lr=swa_lr,
            ),
            Accuracy(true_key="y", pred_key="y_pred"),
        ],
        log_interval=100,
    )

    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic Weight Averaging")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--swa_start", type=int, default=10, help="Epoch to start SWA")
    parser.add_argument("--swa_freq", type=int, default=1, help="SWA update frequency")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--swa_lr", type=float, default=0.001, help="SWA learning rate")
    args = parser.parse_args()

    print(f"Training for {args.epochs} epochs")
    print(f"SWA starts at epoch {args.swa_start} with frequency {args.swa_freq}")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        swa_start=args.swa_start,
        swa_freq=args.swa_freq,
        lr=args.lr,
        swa_lr=args.swa_lr,
    )
    est.fit()

    print("\nTesting with SWA weights...")
    est.test()
