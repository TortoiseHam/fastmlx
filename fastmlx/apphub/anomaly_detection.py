"""Autoencoder-based Anomaly Detection.

This example demonstrates using autoencoders for anomaly detection.

The approach:
1. Train an autoencoder on "normal" data only
2. The autoencoder learns to reconstruct normal patterns well
3. Anomalies have high reconstruction error (they're unlike training data)
4. Use reconstruction error as an anomaly score

Applications:
- Fraud detection
- Manufacturing defect detection
- Network intrusion detection
- Medical anomaly detection

Example usage:
    python anomaly_detection.py --anomaly_digit 9 --threshold_percentile 95
"""

import argparse
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import fastmlx as fe
from fastmlx.op import TensorOp, ModelOp, UpdateOp
from fastmlx.trace import Trace


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for image anomaly detection."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_fc = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)

    def encode(self, x: mx.array) -> mx.array:
        """Encode input to latent space."""
        x = nn.relu(self.enc_conv1(x))
        x = nn.relu(self.enc_conv2(x))
        x = nn.relu(self.enc_conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.enc_fc(x)
        return x

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent vector to image."""
        x = nn.relu(self.dec_fc(z))
        x = x.reshape(-1, 4, 4, 128)
        x = nn.relu(self.dec_conv1(x))
        x = nn.relu(self.dec_conv2(x))
        x = mx.sigmoid(self.dec_conv3(x))

        # Crop to 28x28
        x = x[:, :28, :28, :]
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: encode then decode."""
        z = self.encode(x)
        return self.decode(z)


class ReconstructionLoss(TensorOp):
    """Compute reconstruction loss (MSE)."""

    def __init__(
        self,
        inputs: list[str],
        outputs: str,
        mode: str | list[str] | None = None,
    ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> mx.array:
        """Compute MSE reconstruction loss."""
        x, x_recon = data
        return mx.mean((x - x_recon) ** 2)


class AnomalyScoreOp(TensorOp):
    """Compute per-sample anomaly scores."""

    def __init__(
        self,
        inputs: list[str],
        outputs: str,
        mode: str | list[str] | None = None,
    ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> mx.array:
        """Compute per-sample reconstruction error as anomaly score."""
        x, x_recon = data
        # Per-sample MSE
        scores = mx.mean((x - x_recon) ** 2, axis=(1, 2, 3))
        return scores


class AnomalyEvaluator(Trace):
    """Evaluate anomaly detection performance.

    Args:
        threshold_percentile: Percentile of normal data to use as threshold.
    """

    def __init__(self, threshold_percentile: float = 95):
        super().__init__()
        self.threshold_percentile = threshold_percentile
        self.train_scores = []
        self.test_scores = []
        self.test_labels = []
        self.threshold = None

    def on_batch_end(self, data: dict[str, Any]) -> None:
        """Collect anomaly scores."""
        if "anomaly_score" in data:
            scores = data["anomaly_score"]
            if isinstance(scores, mx.array):
                scores = np.array(scores)

            mode = data.get("mode", "train")

            if mode == "train":
                self.train_scores.extend(scores.flatten().tolist())
            else:
                self.test_scores.extend(scores.flatten().tolist())
                if "is_anomaly" in data:
                    labels = data["is_anomaly"]
                    if isinstance(labels, mx.array):
                        labels = np.array(labels)
                    self.test_labels.extend(labels.flatten().tolist())

    def on_epoch_end(self, data: dict[str, Any]) -> None:
        """Compute threshold and report metrics."""
        mode = data.get("mode", "train")

        if mode == "train" and self.train_scores:
            # Set threshold based on training data
            self.threshold = np.percentile(self.train_scores, self.threshold_percentile)
            data["anomaly_threshold"] = self.threshold
            data["train_mean_score"] = np.mean(self.train_scores)
            self.train_scores = []

        elif mode != "train" and self.test_scores and self.test_labels and self.threshold:
            # Evaluate on test data
            scores = np.array(self.test_scores)
            labels = np.array(self.test_labels)

            # Predictions based on threshold
            predictions = (scores > self.threshold).astype(int)

            # Metrics
            true_positives = np.sum((predictions == 1) & (labels == 1))
            false_positives = np.sum((predictions == 1) & (labels == 0))
            true_negatives = np.sum((predictions == 0) & (labels == 0))
            false_negatives = np.sum((predictions == 0) & (labels == 1))

            accuracy = (true_positives + true_negatives) / len(labels)
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            data["anomaly_accuracy"] = accuracy
            data["anomaly_precision"] = precision
            data["anomaly_recall"] = recall
            data["anomaly_f1"] = f1

            self.test_scores = []
            self.test_labels = []


def get_estimator(
    epochs: int = 20,
    batch_size: int = 128,
    latent_dim: int = 32,
    anomaly_digit: int = 9,
    threshold_percentile: float = 95,
    lr: float = 0.001,
) -> fe.Estimator:
    """Create an estimator for anomaly detection training.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        latent_dim: Dimension of latent space.
        anomaly_digit: Which digit to treat as anomaly (excluded from training).
        threshold_percentile: Percentile for anomaly threshold.
        lr: Learning rate.

    Returns:
        Configured Estimator.
    """
    from fastmlx.dataset.data import mnist

    # Load MNIST dataset
    train_data, test_data = mnist.load_data()
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Create training set without anomaly digit (only "normal" data)
    normal_mask = y_train != anomaly_digit
    x_train_normal = x_train[normal_mask]
    y_train_normal = y_train[normal_mask]

    print(f"Training on {len(x_train_normal)} normal samples (excluding digit {anomaly_digit})")

    # Create test set with anomaly labels
    is_anomaly = (y_test == anomaly_digit).astype(np.float32)
    print(f"Test set: {np.sum(is_anomaly == 0):.0f} normal, {np.sum(is_anomaly == 1):.0f} anomalies")

    # Create pipeline
    pipeline = fe.Pipeline(
        train_data=fe.dataset.NumpyDataset(data={
            "x": x_train_normal,
            "y": y_train_normal,
        }),
        test_data=fe.dataset.NumpyDataset(data={
            "x": x_test,
            "y": y_test,
            "is_anomaly": is_anomaly,
        }),
        batch_size=batch_size,
        ops=[
            fe.op.Normalize(inputs="x", outputs="x", mean=0.0, std=1.0),
            fe.op.ExpandDims(inputs="x", outputs="x", axis=-1),
        ],
    )

    # Build autoencoder
    model = fe.build(
        model=ConvAutoencoder(latent_dim=latent_dim),
        optimizer=optim.Adam(learning_rate=lr),
        model_name="autoencoder",
    )

    network = fe.Network(
        ops=[
            ModelOp(model=model, inputs="x", outputs="x_recon"),
            ReconstructionLoss(
                inputs=["x", "x_recon"],
                outputs="loss",
                mode="train",
            ),
            AnomalyScoreOp(
                inputs=["x", "x_recon"],
                outputs="anomaly_score",
            ),
            UpdateOp(model=model, loss_name="loss"),
        ]
    )

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=[
            AnomalyEvaluator(threshold_percentile=threshold_percentile),
        ],
        log_steps=100,
    )

    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder Anomaly Detection")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--anomaly_digit", type=int, default=9,
                        help="Digit to treat as anomaly (0-9)")
    parser.add_argument("--threshold_percentile", type=float, default=95,
                        help="Percentile for anomaly threshold")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    print(f"Treating digit {args.anomaly_digit} as anomaly")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        anomaly_digit=args.anomaly_digit,
        threshold_percentile=args.threshold_percentile,
        lr=args.lr,
    )
    est.fit()
    est.test()
