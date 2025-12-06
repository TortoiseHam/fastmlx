"""Focal Loss for imbalanced classification using :mod:`fastmlx`.

Demonstrates using Focal Loss to handle class imbalance, where the loss
down-weights easy examples and focuses training on hard negatives.

This is particularly useful for:
- Object detection (many background vs few objects)
- Medical diagnosis (rare diseases)
- Fraud detection (few fraudulent transactions)
- Any dataset with severe class imbalance

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Tuple

import numpy as np
import mlx.core as mx

import fastmlx as fe
from fastmlx.architecture import LeNet
from fastmlx.dataset import MLXDataset
from fastmlx.dataset.data import mnist
from fastmlx.op import Minmax, FocalLoss, CrossEntropy, ModelOp, UpdateOp
from fastmlx.schedule import cosine_decay
from fastmlx.trace.metric import Accuracy, Precision, Recall, F1Score
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.adapt import LRScheduler


def create_imbalanced_mnist(
    imbalance_ratio: float = 0.01,
    minority_classes: list = [8, 9],
    seed: int = 42
) -> Tuple[MLXDataset, MLXDataset]:
    """Create an imbalanced version of MNIST.

    Reduces samples from minority_classes by imbalance_ratio.

    Args:
        imbalance_ratio: Fraction of samples to keep for minority classes.
        minority_classes: List of class indices to undersample.
        seed: Random seed.

    Returns:
        Tuple of (train_data, eval_data) with imbalanced training set.
    """
    train_ds, eval_ds = mnist.load_data()

    np.random.seed(seed)

    # Get data as numpy
    train_x = np.array(train_ds.data["x"])
    train_y = np.array(train_ds.data["y"])

    # Create mask for samples to keep
    keep_mask = np.ones(len(train_y), dtype=bool)

    for cls in minority_classes:
        cls_indices = np.where(train_y == cls)[0]
        n_keep = int(len(cls_indices) * imbalance_ratio)
        remove_indices = np.random.choice(cls_indices, len(cls_indices) - n_keep, replace=False)
        keep_mask[remove_indices] = False

    # Apply mask
    train_x = train_x[keep_mask]
    train_y = train_y[keep_mask]

    # Print class distribution
    print("Training set class distribution:")
    for cls in range(10):
        count = np.sum(train_y == cls)
        print(f"  Class {cls}: {count:5d} samples")

    train_data = MLXDataset({
        "x": mx.array(train_x),
        "y": mx.array(train_y)
    })

    return train_data, eval_ds


def get_estimator(
    epochs: int = 20,
    batch_size: int = 64,
    gamma: float = 2.0,
    alpha: float = 0.25,
    use_focal_loss: bool = True,
    imbalance_ratio: float = 0.01,
    minority_classes: list = [8, 9],
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create focal loss estimator for imbalanced MNIST.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        gamma: Focal loss focusing parameter. Higher = more focus on hard examples.
               gamma=0 is equivalent to cross-entropy.
        alpha: Weighting factor for minority class.
        use_focal_loss: If False, uses standard cross-entropy for comparison.
        imbalance_ratio: Fraction of minority class samples to keep.
        minority_classes: Classes to undersample.
        save_dir: Directory to save best model.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = create_imbalanced_mnist(
        imbalance_ratio=imbalance_ratio,
        minority_classes=minority_classes,
    )

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[Minmax(inputs="x", outputs="x")],
    )

    model = fe.build(
        model_fn=lambda: LeNet(input_shape=(1, 28, 28)),
        optimizer_fn="adam"
    )

    # Choose loss function
    if use_focal_loss:
        loss_op = FocalLoss(
            inputs=("y_pred", "y"),
            outputs="loss",
            gamma=gamma,
            alpha=alpha
        )
        loss_name = "Focal Loss"
    else:
        loss_op = CrossEntropy(inputs=("y_pred", "y"), outputs="loss")
        loss_name = "Cross-Entropy"

    print(f"Using {loss_name}" + (f" (gamma={gamma}, alpha={alpha})" if use_focal_loss else ""))

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        loss_op,
        UpdateOp(model=model, loss_name="loss")
    ])

    steps_per_epoch = len(train_data) // batch_size
    cycle_length = epochs * steps_per_epoch

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        # Track precision/recall for minority classes
        Precision(true_key="y", pred_key="y_pred", average="macro"),
        Recall(true_key="y", pred_key="y_pred", average="macro"),
        F1Score(true_key="y", pred_key="y_pred", average="macro"),
        BestModelSaver(model=model, save_dir=save_dir, metric="f1_score"),
        LRScheduler(
            model=model,
            lr_fn=lambda step: cosine_decay(
                step, cycle_length=cycle_length,
                init_lr=1e-3, min_lr=1e-5
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
    parser = argparse.ArgumentParser(description="Focal Loss for Imbalanced Data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focal loss gamma (focusing parameter)")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Focal loss alpha (class weight)")
    parser.add_argument("--imbalance-ratio", type=float, default=0.01,
                        help="Fraction of minority samples to keep")
    parser.add_argument("--use-cross-entropy", action="store_true",
                        help="Use cross-entropy instead of focal loss")
    args = parser.parse_args()

    print("Focal Loss for Imbalanced Classification")
    print(f"  Imbalance ratio: {args.imbalance_ratio} (minority classes: 8, 9)")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        alpha=args.alpha,
        use_focal_loss=not args.use_cross_entropy,
        imbalance_ratio=args.imbalance_ratio,
    )
    est.fit()
    est.test()
