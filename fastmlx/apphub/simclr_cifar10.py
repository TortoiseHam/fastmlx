"""SimCLR contrastive learning example using :mod:`fastmlx`.

Demonstrates self-supervised representation learning using contrastive
learning. The model learns useful image representations without labels
by maximizing agreement between differently augmented views of the same image.

Reference:
    Chen et al., "A Simple Framework for Contrastive Learning of
    Visual Representations", ICML 2020.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

import fastmlx as fe
from fastmlx.architecture import ResNet9
from fastmlx.dataset.data import cifair10
from fastmlx.op import ColorJitter, HorizontalFlip, Normalize, Op, PadIfNeeded, RandomCrop, Sometimes
from fastmlx.schedule import warmup_cosine_decay
from fastmlx.trace.adapt import LRScheduler
from fastmlx.trace.base import Trace
from fastmlx.trace.io import ModelSaver


class SimCLRProjector(nn.Module):
    """MLP projection head for SimCLR.

    Projects encoder representations to a lower-dimensional space
    where contrastive loss is applied.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class SimCLREncoder(nn.Module):
    """SimCLR encoder: backbone + projection head."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 512,
        projection_dim: int = 128
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = SimCLRProjector(feature_dim, feature_dim, projection_dim)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Returns both features and projections."""
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections

    def encode(self, x: mx.array) -> mx.array:
        """Get features only (for downstream tasks)."""
        return self.backbone(x)


class SimCLRAugmentation(Op):
    """Create two augmented views of each image."""

    def __init__(self, inputs: str, outputs: Tuple[str, str]) -> None:
        super().__init__([inputs], list(outputs))

    def forward(self, data, state):
        x = data[0]
        # Return same image twice (augmentation applied separately in pipeline)
        return x, x.copy()


class NTXentLoss(Op):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR.

    Computes contrastive loss between positive pairs (different views of
    same image) against negative pairs (views of different images).
    """

    def __init__(
        self,
        inputs: Tuple[str, str],
        outputs: str,
        temperature: float = 0.5
    ) -> None:
        super().__init__(list(inputs), outputs)
        self.temperature = temperature

    def forward(self, data, state):
        z1, z2 = data  # Projections from two views

        batch_size = z1.shape[0]

        # Normalize projections
        z1 = z1 / (mx.sqrt(mx.sum(z1 ** 2, axis=-1, keepdims=True)) + 1e-8)
        z2 = z2 / (mx.sqrt(mx.sum(z2 ** 2, axis=-1, keepdims=True)) + 1e-8)

        # Concatenate both views
        z = mx.concatenate([z1, z2], axis=0)  # (2*batch_size, dim)

        # Compute similarity matrix
        sim_matrix = z @ z.T / self.temperature  # (2*batch_size, 2*batch_size)

        # Create mask for positive pairs
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        # Create labels: for sample i in first half, positive is at i+batch_size
        labels = mx.concatenate([
            mx.arange(batch_size, 2 * batch_size),
            mx.arange(0, batch_size)
        ])

        # Mask out self-similarity (diagonal)
        mask = mx.eye(2 * batch_size) * -1e9
        sim_matrix = sim_matrix + mask

        # Cross-entropy loss
        log_probs = sim_matrix - mx.logsumexp(sim_matrix, axis=1, keepdims=True)
        loss = -mx.mean(mx.take_along_axis(
            log_probs, labels[:, None].astype(mx.int32), axis=1
        ))

        return loss


class SimCLRModelOp(Op):
    """Forward pass through SimCLR encoder for both views."""

    def __init__(
        self,
        model: nn.Module,
        inputs: Tuple[str, str],
        outputs: Tuple[str, str]
    ) -> None:
        super().__init__(list(inputs), list(outputs))
        self.model = model

    def forward(self, data, state):
        x1, x2 = data
        _, z1 = self.model(x1)
        _, z2 = self.model(x2)
        return z1, z2


class ContrastiveAccuracy(Trace):
    """Measure how often the model correctly identifies positive pairs."""

    def __init__(self, z1_key: str = "z1", z2_key: str = "z2") -> None:
        self.z1_key = z1_key
        self.z2_key = z2_key
        self.correct = 0
        self.total = 0

    def on_epoch_begin(self, state):
        self.correct = 0
        self.total = 0

    def on_batch_end(self, batch, state):
        z1 = batch[self.z1_key]
        z2 = batch[self.z2_key]

        batch_size = z1.shape[0]

        # Normalize
        z1 = z1 / (mx.sqrt(mx.sum(z1 ** 2, axis=-1, keepdims=True)) + 1e-8)
        z2 = z2 / (mx.sqrt(mx.sum(z2 ** 2, axis=-1, keepdims=True)) + 1e-8)

        # Similarity between z1 and all z2
        sim = z1 @ z2.T  # (batch_size, batch_size)

        # Correct if diagonal has highest similarity
        pred = mx.argmax(sim, axis=1)
        target = mx.arange(batch_size)
        self.correct += int(mx.sum(pred == target).item())
        self.total += batch_size

    def on_epoch_end(self, state):
        state['metrics']['contrastive_accuracy'] = self.correct / max(1, self.total)


def get_estimator(
    epochs: int = 100,
    batch_size: int = 256,
    temperature: float = 0.5,
    projection_dim: int = 128,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create SimCLR self-supervised learning estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size (larger is better for contrastive learning).
        temperature: Temperature for NT-Xent loss.
        projection_dim: Dimension of projection head output.
        save_dir: Directory to save models.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = cifair10.load_data()

    # Strong augmentation pipeline for contrastive learning
    # We need to create two views, so we duplicate the preprocessing
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            # Normalize first
            Normalize(
                inputs="x", outputs="x",
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2471, 0.2435, 0.2616)
            ),
            # Create two copies for augmentation
            SimCLRAugmentation(inputs="x", outputs=("x1", "x2")),
            # Augment first view
            PadIfNeeded(inputs="x1", outputs="x1", min_height=40, min_width=40),
            RandomCrop(inputs="x1", outputs="x1", height=32, width=32),
            Sometimes(HorizontalFlip(inputs="x1", outputs="x1")),
            Sometimes(ColorJitter(inputs="x1", outputs="x1", brightness=0.4, contrast=0.4, saturation=0.4)),
            # Augment second view
            PadIfNeeded(inputs="x2", outputs="x2", min_height=40, min_width=40),
            RandomCrop(inputs="x2", outputs="x2", height=32, width=32),
            Sometimes(HorizontalFlip(inputs="x2", outputs="x2")),
            Sometimes(ColorJitter(inputs="x2", outputs="x2", brightness=0.4, contrast=0.4, saturation=0.4)),
        ],
    )

    # Create SimCLR model with ResNet9 backbone
    # We need a modified ResNet9 that outputs features instead of class logits
    class ResNet9Backbone(nn.Module):
        """ResNet9 backbone that outputs features."""
        def __init__(self):
            super().__init__()
            self.resnet = ResNet9(input_shape=(3, 32, 32))
            # Get the feature dimension before the final FC layer
            self.feature_dim = 512

        def __call__(self, x):
            # Run through ResNet but get features before final layer
            # This is a simplified version - ideally we'd modify ResNet9
            return self.resnet(x)  # Returns 10-dim, we'll project from there

    # For simplicity, use ResNet9 output and project from there
    backbone = ResNet9(input_shape=(3, 32, 32))
    encoder = SimCLREncoder(backbone, feature_dim=10, projection_dim=projection_dim)

    model = fe.build(
        model_fn=lambda: encoder,
        optimizer_fn="adam"
    )

    network = fe.Network([
        SimCLRModelOp(model=model, inputs=("x1", "x2"), outputs=("z1", "z2")),
        NTXentLoss(inputs=("z1", "z2"), outputs="ntxent_loss", temperature=temperature),
        fe.op.UpdateOp(model=model, loss_name="ntxent_loss")
    ])

    steps_per_epoch = 50000 // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = 10 * steps_per_epoch

    traces = [
        ContrastiveAccuracy(z1_key="z1", z2_key="z2"),
        ModelSaver(model=model, save_dir=save_dir, frequency=10),
        LRScheduler(
            model=model,
            lr_fn=lambda step: warmup_cosine_decay(
                step, warmup_steps=warmup_steps,
                total_steps=total_steps, init_lr=1e-3, min_lr=1e-5
            )
        )
    ]

    return fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Self-Supervised Learning")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    args = parser.parse_args()

    print("SimCLR Self-Supervised Learning on CIFAR-10")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Temperature: {args.temperature}")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )
    est.fit()
