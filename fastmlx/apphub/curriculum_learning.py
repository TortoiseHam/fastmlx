"""Curriculum Learning example using :mod:`fastmlx`.

Demonstrates training with curriculum learning, where examples are
presented in order of increasing difficulty. This can lead to:
- Faster convergence
- Better final performance
- More stable training

Difficulty can be measured by:
- Loss on the example
- Confidence of prediction
- Hand-crafted metrics
- Self-paced learning

Reference:
    Bengio et al., "Curriculum Learning", ICML 2009.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Any, MutableMapping, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import fastmlx as fe
from fastmlx.architecture import LeNet
from fastmlx.dataset import MLXDataset
from fastmlx.dataset.data import mnist
from fastmlx.op import Minmax, CrossEntropy, ModelOp, UpdateOp
from fastmlx.schedule import cosine_decay
from fastmlx.trace.metric import Accuracy
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.adapt import LRScheduler


def compute_sample_difficulty(
    model: nn.Module,
    images: mx.array,
    labels: mx.array,
    batch_size: int = 256
) -> np.ndarray:
    """Compute difficulty scores for each sample.

    Uses loss as a proxy for difficulty - higher loss = harder sample.

    Args:
        model: Trained model to evaluate difficulty.
        images: Dataset images.
        labels: Dataset labels.
        batch_size: Batch size for processing.

    Returns:
        Array of difficulty scores (one per sample).
    """
    num_samples = images.shape[0]
    difficulties = np.zeros(num_samples)

    for i in range(0, num_samples, batch_size):
        batch_x = images[i:i + batch_size]
        batch_y = labels[i:i + batch_size]

        # Get predictions
        logits = model(batch_x)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Per-sample cross-entropy loss
        losses = -mx.take_along_axis(
            log_probs, batch_y[:, None].astype(mx.int32), axis=1
        ).flatten()

        difficulties[i:i + batch_size] = np.array(losses)

    return difficulties


def create_curriculum_batches(
    images: mx.array,
    labels: mx.array,
    difficulties: np.ndarray,
    epoch: int,
    total_epochs: int,
    batch_size: int,
    curriculum_type: str = "linear"
) -> Tuple[mx.array, mx.array]:
    """Create training data with curriculum-based sampling.

    Args:
        images: Dataset images.
        labels: Dataset labels.
        difficulties: Difficulty scores per sample.
        epoch: Current epoch.
        total_epochs: Total number of epochs.
        batch_size: Batch size.
        curriculum_type: Type of curriculum:
            - 'linear': Linearly increase difficulty threshold
            - 'exponential': Exponentially increase threshold
            - 'step': Step-wise increase at specific epochs

    Returns:
        Tuple of (selected_images, selected_labels).
    """
    num_samples = len(difficulties)

    # Compute fraction of data to use based on curriculum
    if curriculum_type == "linear":
        # Start with 20% of data, linearly increase to 100%
        fraction = 0.2 + 0.8 * (epoch / total_epochs)
    elif curriculum_type == "exponential":
        # Exponential growth
        fraction = 1 - 0.8 * np.exp(-3 * epoch / total_epochs)
    else:  # step
        # Step-wise: 20% -> 50% -> 100%
        if epoch < total_epochs // 3:
            fraction = 0.2
        elif epoch < 2 * total_epochs // 3:
            fraction = 0.5
        else:
            fraction = 1.0

    # Select easiest samples up to the fraction
    num_select = int(num_samples * fraction)
    sorted_indices = np.argsort(difficulties)
    selected_indices = sorted_indices[:num_select]

    # Shuffle selected samples
    np.random.shuffle(selected_indices)

    return images[selected_indices], labels[selected_indices]


def train_with_curriculum(
    epochs: int = 20,
    batch_size: int = 64,
    curriculum_type: str = "linear",
    pretrain_epochs: int = 2,
    save_dir: str = tempfile.mkdtemp(),
) -> None:
    """Train with curriculum learning.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        curriculum_type: Type of curriculum ('linear', 'exponential', 'step').
        pretrain_epochs: Epochs to pretrain for difficulty estimation.
        save_dir: Directory to save model.
    """
    print(f"Curriculum Learning on MNIST")
    print(f"  Curriculum type: {curriculum_type}")
    print(f"  Pre-train epochs: {pretrain_epochs}")

    # Load data
    train_ds, eval_ds = mnist.load_data()

    train_x = train_ds.data["x"].astype(mx.float32) / 255.0
    train_y = train_ds.data["y"]
    eval_x = eval_ds.data["x"].astype(mx.float32) / 255.0
    eval_y = eval_ds.data["y"]

    num_train = train_x.shape[0]
    num_eval = eval_x.shape[0]

    # Create model
    model = LeNet(input_shape=(1, 28, 28))
    mx.eval(model.parameters())
    optimizer = nn.optimizers.Adam(learning_rate=1e-3)

    def loss_fn(params, x, y):
        model.update(params)
        logits = model(x)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.mean(mx.take_along_axis(
            log_probs, y[:, None].astype(mx.int32), axis=1
        ))

    loss_and_grad = mx.value_and_grad(loss_fn)

    # Pre-train to estimate difficulty
    print(f"\nPhase 1: Pre-training for {pretrain_epochs} epochs to estimate difficulty...")
    for epoch in range(pretrain_epochs):
        indices = mx.random.permutation(num_train)
        for i in range(0, num_train, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]

            loss, grads = loss_and_grad(model.trainable_parameters(), batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

        # Evaluate
        correct = 0
        for i in range(0, num_eval, batch_size):
            logits = model(eval_x[i:i + batch_size])
            preds = mx.argmax(logits, axis=-1)
            correct += int(mx.sum(preds == eval_y[i:i + batch_size]).item())
        print(f"  Pre-train Epoch {epoch + 1}/{pretrain_epochs} - Accuracy: {correct / num_eval:.4f}")

    # Compute difficulty scores
    print("\nComputing sample difficulties...")
    difficulties = compute_sample_difficulty(model, train_x, train_y)
    print(f"  Difficulty range: [{difficulties.min():.4f}, {difficulties.max():.4f}]")
    print(f"  Mean difficulty: {difficulties.mean():.4f}")

    # Main training with curriculum
    print(f"\nPhase 2: Training with {curriculum_type} curriculum for {epochs} epochs...")

    best_acc = 0.0
    for epoch in range(epochs):
        # Get curriculum-based data for this epoch
        curr_x, curr_y = create_curriculum_batches(
            train_x, train_y, difficulties,
            epoch, epochs, batch_size, curriculum_type
        )
        num_curr = curr_x.shape[0]

        # Update learning rate
        progress = epoch / epochs
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * progress)))
        optimizer.learning_rate = lr

        # Train on curriculum data
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, num_curr, batch_size):
            batch_x = curr_x[i:i + batch_size]
            batch_y = curr_y[i:i + batch_size]

            loss, grads = loss_and_grad(model.trainable_parameters(), batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_loss += float(loss.item())
            num_batches += 1

        # Evaluate
        correct = 0
        for i in range(0, num_eval, batch_size):
            logits = model(eval_x[i:i + batch_size])
            preds = mx.argmax(logits, axis=-1)
            correct += int(mx.sum(preds == eval_y[i:i + batch_size]).item())

        acc = correct / num_eval
        avg_loss = epoch_loss / num_batches
        data_fraction = num_curr / num_train

        if acc > best_acc:
            best_acc = acc

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Loss: {avg_loss:.4f}, Acc: {acc:.4f}, "
              f"Data: {data_fraction * 100:.1f}%, LR: {lr:.6f}")

    print(f"\nBest accuracy: {best_acc:.4f}")

    # Save model
    import os
    model_path = os.path.join(save_dir, "curriculum_model.npz")
    model.save_weights(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum Learning with FastMLX")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--curriculum-type", type=str, default="linear",
                        choices=["linear", "exponential", "step"],
                        help="Type of curriculum")
    parser.add_argument("--pretrain-epochs", type=int, default=2,
                        help="Pre-training epochs for difficulty estimation")
    args = parser.parse_args()

    train_with_curriculum(
        epochs=args.epochs,
        batch_size=args.batch_size,
        curriculum_type=args.curriculum_type,
        pretrain_epochs=args.pretrain_epochs,
    )
