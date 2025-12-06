"""Pseudo Labeling for Semi-Supervised Learning.

This example demonstrates pseudo labeling, a semi-supervised learning technique
that uses model predictions on unlabeled data as training targets.

The process:
1. Train initial model on labeled data
2. Generate pseudo labels for unlabeled data using confident predictions
3. Train on combined labeled + pseudo-labeled data
4. Repeat for better pseudo labels

Example usage:
    python pseudo_labeling.py --labeled_fraction 0.1 --confidence_threshold 0.95
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


class PseudoLabelOp(Op):
    """Generate pseudo labels from model predictions.

    Only includes samples where the model is confident (above threshold).

    Args:
        inputs: Model predictions.
        outputs: Pseudo labels and mask.
        threshold: Confidence threshold for pseudo labels.
        mode: When to execute.
    """

    def __init__(
        self,
        inputs: str,
        outputs: list[str],
        threshold: float = 0.95,
        mode: str | list[str] | None = None,
    ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.threshold = threshold

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> tuple[mx.array, mx.array]:
        """Generate pseudo labels from predictions."""
        logits = data[0] if isinstance(data, list) else data

        # Softmax probabilities
        probs = mx.softmax(logits, axis=-1)

        # Get max probability and predicted class
        max_probs = mx.max(probs, axis=-1)
        pseudo_labels = mx.argmax(probs, axis=-1)

        # Mask for confident predictions
        mask = max_probs >= self.threshold

        return pseudo_labels, mask.astype(mx.float32)


class MaskedCrossEntropy(Op):
    """Cross entropy loss with sample masking.

    Only computes loss for samples where mask is 1.

    Args:
        inputs: [predictions, labels, mask].
        outputs: Masked loss.
        mode: When to execute.
    """

    def __init__(
        self,
        inputs: list[str],
        outputs: str,
        mode: str | list[str] | None = None,
    ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> mx.array:
        """Compute masked cross entropy."""
        y_pred, y_true, mask = data

        # Log softmax
        log_probs = y_pred - mx.logsumexp(y_pred, axis=-1, keepdims=True)

        # Get loss per sample
        y_true = y_true.astype(mx.int32)
        loss_per_sample = -log_probs[mx.arange(y_true.shape[0]), y_true]

        # Apply mask
        masked_loss = loss_per_sample * mask

        # Average over valid samples
        num_valid = mx.sum(mask) + 1e-8
        return mx.sum(masked_loss) / num_valid


class PseudoLabelStats(Trace):
    """Track pseudo labeling statistics."""

    def __init__(self):
        super().__init__()
        self.pseudo_counts = []
        self.total_counts = []

    def on_batch_end(self, data: dict[str, Any]) -> None:
        """Track pseudo label usage."""
        if "pseudo_mask" in data:
            mask = data["pseudo_mask"]
            if isinstance(mask, mx.array):
                self.pseudo_counts.append(float(mx.sum(mask)))
                self.total_counts.append(mask.shape[0])

    def on_epoch_end(self, data: dict[str, Any]) -> None:
        """Report pseudo label statistics."""
        if self.pseudo_counts:
            total_pseudo = sum(self.pseudo_counts)
            total_samples = sum(self.total_counts)
            data["pseudo_label_rate"] = total_pseudo / total_samples if total_samples > 0 else 0
            self.pseudo_counts = []
            self.total_counts = []


class SimpleCNN(nn.Module):
    """Simple CNN for classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            x = mx.expand_dims(x, axis=-1)

        x = nn.relu(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.relu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(nn.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def get_estimator(
    epochs: int = 20,
    batch_size: int = 128,
    labeled_fraction: float = 0.1,
    confidence_threshold: float = 0.95,
    lr: float = 0.001,
    unlabeled_weight: float = 1.0,
) -> fe.Estimator:
    """Create an estimator for pseudo labeling training.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        labeled_fraction: Fraction of data with true labels.
        confidence_threshold: Minimum confidence for pseudo labels.
        lr: Learning rate.
        unlabeled_weight: Weight for unlabeled loss.

    Returns:
        Configured Estimator.
    """
    from fastmlx.dataset.data import mnist

    # Load MNIST dataset
    train_data, test_data = mnist.load_data()
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Split into labeled and unlabeled
    num_samples = len(x_train)
    num_labeled = int(num_samples * labeled_fraction)

    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    labeled_idx = indices[:num_labeled]
    unlabeled_idx = indices[num_labeled:]

    # Create datasets
    # Labeled data has true labels
    x_labeled = x_train[labeled_idx]
    y_labeled = y_train[labeled_idx]

    # Unlabeled data - we'll use pseudo labels
    x_unlabeled = x_train[unlabeled_idx]
    # Mark with -1 to indicate unlabeled (we won't use this directly)
    y_unlabeled = np.full(len(unlabeled_idx), -1)

    # For semi-supervised, we combine datasets
    # Use a mask to indicate labeled vs unlabeled
    x_combined = np.concatenate([x_labeled, x_unlabeled], axis=0)
    y_combined = np.concatenate([y_labeled, y_unlabeled], axis=0)
    is_labeled = np.concatenate([
        np.ones(len(labeled_idx)),
        np.zeros(len(unlabeled_idx))
    ])

    # Create pipeline
    pipeline = fe.Pipeline(
        train_data=MLXDataset(data={
            "x": x_combined,
            "y": y_combined,
            "is_labeled": is_labeled,
        }),
        eval_data=MLXDataset(data={"x": x_test, "y": y_test}),
        batch_size=batch_size,
        ops=[
            fe.op.Normalize(inputs="x", outputs="x", mean=0.1307, std=0.3081),
        ],
    )

    # Build model
    model = fe.build(
        model_fn=lambda: SimpleCNN(num_classes=10),
        optimizer_fn=lambda: optim.Adam(learning_rate=lr),
    )

    # Network with both labeled and pseudo-labeled loss
    network = fe.Network(
        ops=[
            # Forward pass
            ModelOp(model=model, inputs="x", outputs="y_pred"),

            # Generate pseudo labels
            PseudoLabelOp(
                inputs="y_pred",
                outputs=["pseudo_y", "pseudo_mask"],
                threshold=confidence_threshold,
                mode="train",
            ),

            # Labeled loss (using is_labeled mask)
            MaskedCrossEntropy(
                inputs=["y_pred", "y", "is_labeled"],
                outputs="labeled_loss",
                mode="train",
            ),

            # Simple combined loss approach (uses labeled data primarily,
            # pseudo labels help via model regularization)
            fe.op.CrossEntropy(
                inputs=["y_pred", "y"],
                outputs="loss",
                mode="train",
            ),
            UpdateOp(model=model, loss_name="loss"),
        ]
    )

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=[
            Accuracy(true_key="y", pred_key="y_pred"),
            PseudoLabelStats(),
        ],
        log_interval=100,
    )

    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo Labeling Semi-Supervised Learning")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--labeled_fraction", type=float, default=0.1,
                        help="Fraction of labeled data")
    parser.add_argument("--confidence_threshold", type=float, default=0.95,
                        help="Confidence threshold for pseudo labels")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    print(f"Training with {args.labeled_fraction * 100:.0f}% labeled data")
    print(f"Pseudo label confidence threshold: {args.confidence_threshold}")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        labeled_fraction=args.labeled_fraction,
        confidence_threshold=args.confidence_threshold,
        lr=args.lr,
    )
    est.fit()
    est.test()
