"""Siamese Network for one-shot learning on MNIST using :mod:`fastmlx`.

Demonstrates learning similarity between image pairs using contrastive loss.
After training, the network can determine if two images belong to the same
class without retraining, enabling one-shot/few-shot learning.

Reference:
    Koch et al., "Siamese Neural Networks for One-shot Image Recognition", 2015.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Tuple

import mlx.core as mx
import numpy as np

import fastmlx as fe
from fastmlx.architecture import SiameseEncoder
from fastmlx.dataset import MLXDataset
from fastmlx.dataset.data import mnist
from fastmlx.op import ContrastiveLoss, Minmax, Op, UpdateOp
from fastmlx.schedule import cosine_decay
from fastmlx.trace.adapt import LRScheduler
from fastmlx.trace.base import Trace
from fastmlx.trace.io import ModelSaver


class SiameseModelOp(Op):
    """Model operation for Siamese network that processes pairs.

    Takes two inputs and produces two embeddings using a shared encoder.
    """

    def __init__(
        self,
        model,
        inputs: Tuple[str, str],
        outputs: Tuple[str, str]
    ) -> None:
        super().__init__(list(inputs), list(outputs))
        self.model = model

    def forward(self, data, state):
        x1, x2 = data
        emb1 = self.model(x1)
        emb2 = self.model(x2)
        return emb1, emb2


class PairAccuracy(Trace):
    """Compute accuracy for pair similarity predictions.

    Predicts pairs as similar if L2 distance < threshold.
    """

    def __init__(
        self,
        emb1_key: str = "emb1",
        emb2_key: str = "emb2",
        label_key: str = "is_similar",
        threshold: float = 1.0,
        output_name: str = "pair_accuracy"
    ) -> None:
        self.emb1_key = emb1_key
        self.emb2_key = emb2_key
        self.label_key = label_key
        self.threshold = threshold
        self.output_name = output_name
        self.correct = 0
        self.total = 0

    def on_epoch_begin(self, state):
        self.correct = 0
        self.total = 0

    def on_batch_end(self, batch, state):
        emb1 = batch[self.emb1_key]
        emb2 = batch[self.emb2_key]
        labels = batch[self.label_key]

        # Calculate L2 distance
        dist = mx.sqrt(mx.sum((emb1 - emb2) ** 2, axis=-1))
        # Predict similar if distance < threshold
        pred_similar = (dist < self.threshold).astype(mx.int32)
        true_labels = labels.astype(mx.int32).flatten()

        self.correct += int(mx.sum(pred_similar == true_labels).item())
        self.total += true_labels.shape[0]

    def on_epoch_end(self, state):
        state['metrics'][self.output_name] = self.correct / max(1, self.total)


def create_pairs(
    images: np.ndarray,
    labels: np.ndarray,
    num_pairs: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create positive and negative pairs from images.

    Args:
        images: Array of images (N, C, H, W).
        labels: Array of labels (N,).
        num_pairs: Number of pairs to create.
        seed: Random seed.

    Returns:
        Tuple of (pairs1, pairs2, is_similar) where is_similar is 1 for
        same-class pairs and 0 for different-class pairs.
    """
    np.random.seed(seed)

    pairs1 = []
    pairs2 = []
    is_similar = []

    # Group images by class
    unique_labels = np.unique(labels)
    class_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    for _ in range(num_pairs):
        # 50% positive pairs, 50% negative pairs
        if np.random.random() > 0.5:
            # Positive pair (same class)
            label = np.random.choice(unique_labels)
            indices = class_indices[label]
            if len(indices) >= 2:
                idx1, idx2 = np.random.choice(indices, 2, replace=False)
            else:
                idx1 = idx2 = indices[0]
            pairs1.append(images[idx1])
            pairs2.append(images[idx2])
            is_similar.append(1)
        else:
            # Negative pair (different classes)
            label1, label2 = np.random.choice(unique_labels, 2, replace=False)
            idx1 = np.random.choice(class_indices[label1])
            idx2 = np.random.choice(class_indices[label2])
            pairs1.append(images[idx1])
            pairs2.append(images[idx2])
            is_similar.append(0)

    return (
        np.array(pairs1),
        np.array(pairs2),
        np.array(is_similar, dtype=np.float32)
    )


def get_estimator(
    epochs: int = 10,
    batch_size: int = 64,
    num_train_pairs: int = 50000,
    num_eval_pairs: int = 10000,
    embedding_dim: int = 128,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create Siamese network estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        num_train_pairs: Number of training pairs to generate.
        num_eval_pairs: Number of evaluation pairs to generate.
        embedding_dim: Dimension of the embedding space.
        save_dir: Directory to save best model.

    Returns:
        Configured Estimator ready for training.
    """
    # Load MNIST
    train_ds, eval_ds = mnist.load_data()

    # Convert to numpy for pair creation
    train_images = np.array(train_ds.data["x"])
    train_labels = np.array(train_ds.data["y"])
    eval_images = np.array(eval_ds.data["x"])
    eval_labels = np.array(eval_ds.data["y"])

    # Create pairs
    train_x1, train_x2, train_similar = create_pairs(
        train_images, train_labels, num_train_pairs, seed=42
    )
    eval_x1, eval_x2, eval_similar = create_pairs(
        eval_images, eval_labels, num_eval_pairs, seed=123
    )

    # Create datasets
    train_data = MLXDataset({
        "x1": mx.array(train_x1),
        "x2": mx.array(train_x2),
        "is_similar": mx.array(train_similar)
    })
    eval_data = MLXDataset({
        "x1": mx.array(eval_x1),
        "x2": mx.array(eval_x2),
        "is_similar": mx.array(eval_similar)
    })

    # Pipeline
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Minmax(inputs="x1", outputs="x1"),
            Minmax(inputs="x2", outputs="x2"),
        ],
    )

    # Build Siamese encoder (shared weights)
    model = fe.build(
        model_fn=lambda: SiameseEncoder(
            input_shape=(1, 28, 28),
            embedding_dim=embedding_dim
        ),
        optimizer_fn="adam"
    )

    # Network
    network = fe.Network([
        SiameseModelOp(model=model, inputs=("x1", "x2"), outputs=("emb1", "emb2")),
        ContrastiveLoss(
            inputs=("emb1", "emb2", "is_similar"),
            outputs="contrastive_loss",
            margin=2.0
        ),
        UpdateOp(model=model, loss_name="contrastive_loss")
    ])

    # Traces
    steps_per_epoch = num_train_pairs // batch_size
    cycle_length = epochs * steps_per_epoch

    traces = [
        PairAccuracy(
            emb1_key="emb1",
            emb2_key="emb2",
            label_key="is_similar",
            threshold=1.0
        ),
        ModelSaver(model=model, save_dir=save_dir, frequency=5),
        LRScheduler(
            model=model,
            lr_fn=lambda step: cosine_decay(
                step,
                cycle_length=cycle_length,
                init_lr=1e-3,
                min_lr=1e-5
            )
        )
    ]

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )
    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Siamese Network One-Shot Learning with FastMLX"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Embedding dimension")
    args = parser.parse_args()

    print("Siamese Network for One-Shot Learning")
    print(f"  Embedding dimension: {args.embedding_dim}")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
    )
    est.fit()
    est.test()
