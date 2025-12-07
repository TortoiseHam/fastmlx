"""Early Stopping example using :mod:`fastmlx`.

Demonstrates the use of EarlyStopping and other adaptive training traces
to automatically stop training when the model stops improving, reducing
wasted computation and preventing overfitting.

This example shows:
- EarlyStopping: Stop when validation metric plateaus
- ReduceLROnPlateau: Lower learning rate when progress stalls
- TerminateOnNaN: Stop immediately if NaN loss is detected
"""

from __future__ import annotations

import argparse
import tempfile

import fastmlx as fe
from fastmlx.architecture import LeNet
from fastmlx.dataset.data import mnist
from fastmlx.op import CrossEntropy, Minmax, ModelOp, UpdateOp
from fastmlx.trace.adapt import (
    EarlyStopping,
    ReduceLROnPlateau,
    TerminateOnNaN,
)
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.metric import Accuracy


def get_estimator(
    max_epochs: int = 100,
    batch_size: int = 64,
    patience: int = 10,
    min_delta: float = 0.001,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create estimator with early stopping.

    This example trains for up to max_epochs but will stop early if:
    1. Validation accuracy doesn't improve for `patience` epochs
    2. NaN loss is detected

    Learning rate is also reduced when progress stalls.

    Args:
        max_epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        lr_patience: Epochs to wait before reducing learning rate.
        lr_factor: Factor to reduce learning rate by.
        save_dir: Directory to save best model.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = mnist.load_data()

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

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    traces = [
        # Metrics
        Accuracy(true_key="y", pred_key="y_pred"),

        # Early stopping based on validation accuracy
        EarlyStopping(
            monitor="accuracy",
            patience=patience,
            min_delta=min_delta,
            compare_mode="max",  # Higher accuracy is better
        ),

        # Reduce learning rate when accuracy plateaus
        ReduceLROnPlateau(
            model=model,
            monitor="accuracy",
            factor=lr_factor,
            patience=lr_patience,
            min_delta=min_delta,
            compare_mode="max",
            min_lr=1e-6,
        ),

        # Stop if NaN loss is detected
        TerminateOnNaN(monitor="ce"),

        # Save best model based on accuracy
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy"),
    ]

    return fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=max_epochs,
        traces=traces
    )


def get_estimator_loss_based(
    max_epochs: int = 100,
    batch_size: int = 64,
    patience: int = 15,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create estimator with loss-based early stopping.

    Alternative configuration that monitors loss instead of accuracy.
    This can be useful when:
    - You don't have a clear accuracy metric
    - You want to detect overfitting earlier
    - Training on regression tasks

    Args:
        max_epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        patience: Number of epochs to wait for improvement.
        save_dir: Directory to save best model.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = mnist.load_data()

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

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),

        # Monitor loss (lower is better)
        EarlyStopping(
            monitor="ce",
            patience=patience,
            min_delta=0.0001,
            compare_mode="min",  # Lower loss is better
        ),

        ReduceLROnPlateau(
            model=model,
            monitor="ce",
            factor=0.5,
            patience=5,
            compare_mode="min",
        ),

        TerminateOnNaN(monitor="ce"),
        BestModelSaver(model=model, save_dir=save_dir, metric="ce"),
    ]

    return fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=max_epochs,
        traces=traces
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early Stopping Example with FastMLX")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--monitor", type=str, choices=["accuracy", "loss"],
                        default="accuracy", help="Metric to monitor")
    args = parser.parse_args()

    print("Early Stopping Training Example")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Patience: {args.patience}")
    print(f"  Monitoring: {args.monitor}")

    if args.monitor == "accuracy":
        est = get_estimator(
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
    else:
        est = get_estimator_loss_based(
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )

    est.fit()
    est.test()
