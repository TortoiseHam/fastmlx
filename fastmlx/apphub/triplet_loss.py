"""Triplet Loss for Metric Learning.

This example demonstrates triplet loss for learning embeddings where similar
items are close together and dissimilar items are far apart.

Triplet loss uses three samples:
- Anchor: A reference sample
- Positive: A sample similar to anchor (same class)
- Negative: A sample dissimilar to anchor (different class)

Loss: max(0, d(anchor, positive) - d(anchor, negative) + margin)

This is widely used for:
- Face recognition/verification
- Image retrieval
- Recommendation systems
- One-shot learning

Example usage:
    python triplet_loss.py --margin 0.5 --embedding_dim 64
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


class TripletMiningOp(TensorOp):
    """Mine triplets from a batch of samples.

    Uses semi-hard negative mining: selects negatives that are farther than
    the positive but within the margin.

    Args:
        inputs: [embeddings, labels].
        outputs: [anchor_idx, positive_idx, negative_idx].
        mode: When to execute.
    """

    def __init__(
        self,
        inputs: list[str],
        outputs: list[str],
        mode: str | list[str] | None = None,
    ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> tuple[mx.array, mx.array, mx.array]:
        """Mine triplets from batch."""
        embeddings, labels = data
        batch_size = embeddings.shape[0]

        # Compute pairwise distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        sq_norms = mx.sum(embeddings ** 2, axis=1, keepdims=True)
        distances = sq_norms + sq_norms.T - 2 * embeddings @ embeddings.T
        distances = mx.maximum(distances, 0.0)  # Numerical stability

        # Create mask for valid pairs
        labels = labels.flatten()
        same_class = labels[:, None] == labels[None, :]
        diff_class = ~same_class

        # For each anchor, find hardest positive and semi-hard negative
        anchors = []
        positives = []
        negatives = []

        # Convert to numpy for indexing
        labels_np = np.array(labels)
        distances_np = np.array(distances)

        for i in range(batch_size):
            # Find positives (same class, not self)
            pos_mask = (labels_np == labels_np[i]) & (np.arange(batch_size) != i)
            pos_indices = np.where(pos_mask)[0]

            if len(pos_indices) == 0:
                continue

            # Find negatives (different class)
            neg_mask = labels_np != labels_np[i]
            neg_indices = np.where(neg_mask)[0]

            if len(neg_indices) == 0:
                continue

            # Hardest positive (furthest)
            pos_dists = distances_np[i, pos_indices]
            hardest_pos = pos_indices[np.argmax(pos_dists)]

            # Semi-hard negative (closest that is still farther than positive)
            pos_dist = distances_np[i, hardest_pos]
            neg_dists = distances_np[i, neg_indices]

            # Find negatives farther than positive
            valid_negs = neg_dists > pos_dist
            if np.any(valid_negs):
                valid_neg_indices = neg_indices[valid_negs]
                valid_neg_dists = neg_dists[valid_negs]
                hardest_neg = valid_neg_indices[np.argmin(valid_neg_dists)]
            else:
                # Fall back to hardest negative
                hardest_neg = neg_indices[np.argmin(neg_dists)]

            anchors.append(i)
            positives.append(hardest_pos)
            negatives.append(hardest_neg)

        if len(anchors) == 0:
            # No valid triplets found
            return mx.array([0]), mx.array([0]), mx.array([1] if batch_size > 1 else [0])

        return (
            mx.array(anchors),
            mx.array(positives),
            mx.array(negatives),
        )


class TripletLoss(TensorOp):
    """Compute triplet loss.

    Args:
        inputs: [embeddings, anchor_idx, positive_idx, negative_idx].
        outputs: Triplet loss.
        margin: Margin for triplet loss.
        mode: When to execute.
    """

    def __init__(
        self,
        inputs: list[str],
        outputs: str,
        margin: float = 0.5,
        mode: str | list[str] | None = None,
    ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.margin = margin

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> mx.array:
        """Compute triplet loss."""
        embeddings, anchor_idx, pos_idx, neg_idx = data

        # Get embeddings for triplets
        anchors = embeddings[anchor_idx.astype(mx.int32)]
        positives = embeddings[pos_idx.astype(mx.int32)]
        negatives = embeddings[neg_idx.astype(mx.int32)]

        # Compute distances
        pos_dist = mx.sum((anchors - positives) ** 2, axis=1)
        neg_dist = mx.sum((anchors - negatives) ** 2, axis=1)

        # Triplet loss with margin
        loss = mx.maximum(pos_dist - neg_dist + self.margin, 0.0)

        return mx.mean(loss)


class EmbeddingNetwork(nn.Module):
    """Network that produces normalized embeddings."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            x = mx.expand_dims(x, axis=-1)

        x = nn.relu(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = nn.relu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = nn.relu(self.conv3(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)

        # L2 normalize embeddings
        x = x / (mx.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        return x


class EmbeddingQualityTrace(Trace):
    """Monitor embedding quality during training."""

    def __init__(self):
        super().__init__()
        self.intra_class_dists = []
        self.inter_class_dists = []

    def on_batch_end(self, data: dict[str, Any]) -> None:
        """Compute embedding statistics."""
        if "embeddings" in data and "y" in data:
            emb = data["embeddings"]
            labels = data["y"]

            if isinstance(emb, mx.array) and isinstance(labels, mx.array):
                emb_np = np.array(emb)
                labels_np = np.array(labels).flatten()

                # Sample pairs for efficiency
                n = min(100, len(emb_np))
                idx = np.random.choice(len(emb_np), n, replace=False)

                intra = []
                inter = []

                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.sum((emb_np[idx[i]] - emb_np[idx[j]]) ** 2)
                        if labels_np[idx[i]] == labels_np[idx[j]]:
                            intra.append(dist)
                        else:
                            inter.append(dist)

                if intra:
                    self.intra_class_dists.extend(intra)
                if inter:
                    self.inter_class_dists.extend(inter)

    def on_epoch_end(self, data: dict[str, Any]) -> None:
        """Report embedding quality metrics."""
        if self.intra_class_dists and self.inter_class_dists:
            avg_intra = np.mean(self.intra_class_dists)
            avg_inter = np.mean(self.inter_class_dists)

            data["avg_intra_dist"] = avg_intra
            data["avg_inter_dist"] = avg_inter
            data["dist_ratio"] = avg_inter / (avg_intra + 1e-8)

            self.intra_class_dists = []
            self.inter_class_dists = []


def get_estimator(
    epochs: int = 20,
    batch_size: int = 128,
    embedding_dim: int = 64,
    margin: float = 0.5,
    lr: float = 0.001,
) -> fe.Estimator:
    """Create an estimator for triplet loss training.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size (should be large for triplet mining).
        embedding_dim: Dimension of embeddings.
        margin: Triplet loss margin.
        lr: Learning rate.

    Returns:
        Configured Estimator.
    """
    from fastmlx.dataset.data import mnist

    # Load MNIST dataset
    train_data, test_data = mnist.load_data()

    # Create pipeline with larger batch for triplet mining
    pipeline = fe.Pipeline(
        train_data=fe.dataset.NumpyDataset(data={"x": train_data[0], "y": train_data[1]}),
        test_data=fe.dataset.NumpyDataset(data={"x": test_data[0], "y": test_data[1]}),
        batch_size=batch_size,
        ops=[
            fe.op.Normalize(inputs="x", outputs="x", mean=0.1307, std=0.3081),
        ],
    )

    # Build model
    model = fe.build(
        model=EmbeddingNetwork(embedding_dim=embedding_dim),
        optimizer=optim.Adam(learning_rate=lr),
        model_name="triplet_net",
    )

    network = fe.Network(
        ops=[
            ModelOp(model=model, inputs="x", outputs="embeddings"),
            TripletMiningOp(
                inputs=["embeddings", "y"],
                outputs=["anchor_idx", "pos_idx", "neg_idx"],
                mode="train",
            ),
            TripletLoss(
                inputs=["embeddings", "anchor_idx", "pos_idx", "neg_idx"],
                outputs="loss",
                margin=margin,
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
            EmbeddingQualityTrace(),
        ],
        log_steps=100,
    )

    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triplet Loss Metric Learning")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--margin", type=float, default=0.5, help="Triplet loss margin")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        lr=args.lr,
    )
    est.fit()
    est.test()
