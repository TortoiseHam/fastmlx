"""Deep Neural Network for Tabular Data using :mod:`fastmlx`.

Demonstrates training a fully-connected neural network on tabular
(structured) data. Uses synthetic data for demonstration, but the
pattern applies to real datasets like housing prices, customer churn,
credit scoring, etc.

This example shows:
- Custom MLP architecture for tabular data
- Handling of mixed feature types
- Proper normalization for tabular features
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import fastmlx as fe
from fastmlx.dataset import MLXDataset
from fastmlx.op import Normalize, MeanSquaredError, CrossEntropy, ModelOp, UpdateOp, Op
from fastmlx.schedule import cosine_decay
from fastmlx.trace.base import Trace
from fastmlx.trace.metric import Accuracy
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.adapt import LRScheduler


class TabularMLP(nn.Module):
    """Multi-layer perceptron for tabular data.

    Architecture designed for structured data with:
    - Batch normalization for stable training
    - Dropout for regularization
    - LeakyReLU activations

    Args:
        input_dim: Number of input features.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Number of output classes/values.
        dropout: Dropout probability.
        task: 'classification' or 'regression'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        output_dim: int = 1,
        dropout: float = 0.2,
        task: str = "classification"
    ) -> None:
        super().__init__()
        self.task = task

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        features = self.backbone(x)
        output = self.head(features)

        if self.task == "regression":
            return output.squeeze(-1)
        return output


class R2Score(Trace):
    """Compute R-squared (coefficient of determination) for regression.

    R2 = 1 - SS_res / SS_tot
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        output_name: str = "r2_score"
    ) -> None:
        self.true_key = true_key
        self.pred_key = pred_key
        self.output_name = output_name
        self.ss_res = 0.0
        self.ss_tot = 0.0
        self.y_sum = 0.0
        self.count = 0

    def on_epoch_begin(self, state):
        self.ss_res = 0.0
        self.ss_tot = 0.0
        self.y_sum = 0.0
        self.count = 0
        self._all_y = []
        self._all_pred = []

    def on_batch_end(self, batch, state):
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]

        if y.ndim > 1:
            y = y.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        self._all_y.append(y)
        self._all_pred.append(y_pred)

    def on_epoch_end(self, state):
        y = mx.concatenate(self._all_y)
        y_pred = mx.concatenate(self._all_pred)

        y_mean = mx.mean(y)
        ss_tot = mx.sum((y - y_mean) ** 2)
        ss_res = mx.sum((y - y_pred) ** 2)

        r2 = 1 - float(ss_res.item()) / max(float(ss_tot.item()), 1e-8)
        state['metrics'][self.output_name] = r2


def generate_classification_data(
    num_samples: int = 10000,
    num_features: int = 20,
    num_classes: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data.

    Creates data with distinct class clusters for demonstration.
    """
    np.random.seed(seed)

    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.zeros(num_samples, dtype=np.int32)

    # Create class-specific patterns
    for c in range(num_classes):
        mask = np.random.rand(num_samples) < (1.0 / num_classes)
        if c > 0:
            mask = mask & (y == 0)
        y[mask] = c

        # Add class-specific signal to first few features
        X[mask, :5] += np.random.randn(5) * 2

    return X, y


def generate_regression_data(
    num_samples: int = 10000,
    num_features: int = 20,
    noise_level: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    Creates data with polynomial relationships for demonstration.
    """
    np.random.seed(seed)

    X = np.random.randn(num_samples, num_features).astype(np.float32)

    # Create target as combination of features with some nonlinearity
    weights = np.random.randn(num_features)
    y = X @ weights
    y += 0.5 * X[:, 0] ** 2  # Add nonlinearity
    y += 0.3 * X[:, 1] * X[:, 2]  # Add interaction
    y += noise_level * np.random.randn(num_samples)
    y = y.astype(np.float32)

    return X, y


def get_classification_estimator(
    epochs: int = 20,
    batch_size: int = 64,
    num_features: int = 20,
    num_classes: int = 5,
    hidden_dims: list = [256, 128, 64],
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create tabular classification estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        num_features: Number of input features.
        num_classes: Number of output classes.
        hidden_dims: Hidden layer dimensions.
        save_dir: Directory to save best model.

    Returns:
        Configured Estimator ready for training.
    """
    # Generate synthetic data
    X_train, y_train = generate_classification_data(
        num_samples=8000, num_features=num_features, num_classes=num_classes, seed=42
    )
    X_eval, y_eval = generate_classification_data(
        num_samples=2000, num_features=num_features, num_classes=num_classes, seed=123
    )

    # Calculate normalization statistics
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    train_data = MLXDataset({
        "x": mx.array(X_train),
        "y": mx.array(y_train)
    })
    eval_data = MLXDataset({
        "x": mx.array(X_eval),
        "y": mx.array(y_eval)
    })

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(
                inputs="x", outputs="x",
                mean=tuple(mean.tolist()),
                std=tuple(std.tolist())
            ),
        ],
    )

    model = fe.build(
        model_fn=lambda: TabularMLP(
            input_dim=num_features,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            task="classification"
        ),
        optimizer_fn="adam"
    )

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    steps_per_epoch = 8000 // batch_size
    cycle_length = epochs * steps_per_epoch

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy"),
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


def get_regression_estimator(
    epochs: int = 20,
    batch_size: int = 64,
    num_features: int = 20,
    hidden_dims: list = [256, 128, 64],
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create tabular regression estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        num_features: Number of input features.
        hidden_dims: Hidden layer dimensions.
        save_dir: Directory to save best model.

    Returns:
        Configured Estimator ready for training.
    """
    # Generate synthetic data
    X_train, y_train = generate_regression_data(
        num_samples=8000, num_features=num_features, seed=42
    )
    X_eval, y_eval = generate_regression_data(
        num_samples=2000, num_features=num_features, seed=123
    )

    # Calculate normalization statistics
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    train_data = MLXDataset({
        "x": mx.array(X_train),
        "y": mx.array(y_train)
    })
    eval_data = MLXDataset({
        "x": mx.array(X_eval),
        "y": mx.array(y_eval)
    })

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(
                inputs="x", outputs="x",
                mean=tuple(mean.tolist()),
                std=tuple(std.tolist())
            ),
        ],
    )

    model = fe.build(
        model_fn=lambda: TabularMLP(
            input_dim=num_features,
            hidden_dims=hidden_dims,
            output_dim=1,
            task="regression"
        ),
        optimizer_fn="adam"
    )

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        MeanSquaredError(inputs=("y_pred", "y"), outputs="mse"),
        UpdateOp(model=model, loss_name="mse")
    ])

    steps_per_epoch = 8000 // batch_size
    cycle_length = epochs * steps_per_epoch

    traces = [
        R2Score(true_key="y", pred_key="y_pred"),
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
    parser = argparse.ArgumentParser(
        description="Tabular DNN with FastMLX"
    )
    parser.add_argument("--task", type=str, choices=["classification", "regression"],
                        default="classification", help="Task type")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-features", type=int, default=20,
                        help="Number of input features")
    parser.add_argument("--num-classes", type=int, default=5,
                        help="Number of classes (classification only)")
    args = parser.parse_args()

    print(f"Tabular DNN - {args.task.capitalize()}")
    print(f"  Features: {args.num_features}")
    if args.task == "classification":
        print(f"  Classes: {args.num_classes}")

    if args.task == "classification":
        est = get_classification_estimator(
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_features=args.num_features,
            num_classes=args.num_classes,
        )
    else:
        est = get_regression_estimator(
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_features=args.num_features,
        )

    est.fit()
    est.test()
