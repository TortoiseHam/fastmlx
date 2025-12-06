"""Multi-task Learning example using :mod:`fastmlx`.

Demonstrates training a single model on multiple tasks simultaneously
using uncertainty-weighted losses. This approach learns task weights
automatically based on homoscedastic uncertainty.

Benefits:
- Shared representations improve generalization
- Efficient inference (one model, multiple outputs)
- Automatic task balancing

Reference:
    Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics", CVPR 2018.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Any, MutableMapping, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import fastmlx as fe
from fastmlx.dataset import MLXDataset
from fastmlx.op import Minmax, Op
from fastmlx.schedule import cosine_decay
from fastmlx.trace.base import Trace
from fastmlx.trace.io import ModelSaver
from fastmlx.trace.adapt import LRScheduler


class MultiTaskModel(nn.Module):
    """Multi-task model with shared encoder and task-specific heads.

    Example tasks on MNIST:
    - Task 1: Digit classification (10 classes)
    - Task 2: Odd/Even classification (2 classes)

    Args:
        input_shape: Input shape as (channels, height, width).
        task_outputs: Dictionary mapping task names to output dimensions.
        shared_dim: Dimension of shared representation.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        task_outputs: dict = {"digit": 10, "odd_even": 2},
        shared_dim: int = 256
    ) -> None:
        super().__init__()

        in_channels = input_shape[0]

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate flattened size
        h, w = input_shape[1] // 8, input_shape[2] // 8
        flat_size = 128 * max(1, h) * max(1, w)

        # Shared FC layer
        self.shared_fc = nn.Sequential(
            nn.Linear(flat_size, shared_dim),
            nn.ReLU(),
        )

        # Task-specific heads
        self.task_heads = {}
        for task_name, num_outputs in task_outputs.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(shared_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_outputs),
            )

    def __call__(self, x: mx.array) -> dict:
        """Forward pass returning predictions for all tasks.

        Args:
            x: Input of shape (batch, height, width, channels).

        Returns:
            Dictionary mapping task names to predictions.
        """
        # Shared encoding
        features = self.encoder(x)
        features = features.reshape(features.shape[0], -1)
        shared = self.shared_fc(features)

        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared)

        return outputs


class MultiTaskModelOp(Op):
    """Forward pass through multi-task model."""

    def __init__(self, model, inputs: str, outputs: list) -> None:
        super().__init__([inputs], outputs)
        self.model = model

    def forward(self, data, state):
        x = data[0]
        task_outputs = self.model(x)
        # Return outputs in order specified
        return tuple(task_outputs[key] for key in self.outputs)


class UncertaintyWeightedLoss(Op):
    """Uncertainty-weighted multi-task loss.

    Learns task weights automatically based on homoscedastic uncertainty.
    Each task has a learnable log variance parameter.

    Loss = sum_t ( (1/2) * exp(-s_t) * L_t + (1/2) * s_t )

    where s_t is the log variance for task t.
    """

    def __init__(
        self,
        inputs: list,
        outputs: str,
        task_names: list,
        num_classes: dict
    ) -> None:
        """Initialize uncertainty-weighted loss.

        Args:
            inputs: List of (prediction, target) pairs for each task.
            outputs: Output key for total loss.
            task_names: List of task names.
            num_classes: Dictionary mapping task names to number of classes.
        """
        super().__init__(inputs, outputs)
        self.task_names = task_names
        self.num_classes = num_classes

        # Learnable log variance for each task (initialized to 0)
        self.log_vars = {name: mx.array(0.0) for name in task_names}

    def forward(self, data, state):
        # Data is flattened list: [pred1, target1, pred2, target2, ...]
        total_loss = mx.array(0.0)

        for i, task_name in enumerate(self.task_names):
            pred = data[i * 2]
            target = data[i * 2 + 1]

            # Cross-entropy loss for this task
            log_probs = pred - mx.logsumexp(pred, axis=-1, keepdims=True)
            if target.ndim > 1:
                # One-hot
                task_loss = -mx.mean(mx.sum(target * log_probs, axis=-1))
            else:
                # Integer labels
                task_loss = -mx.mean(mx.take_along_axis(
                    log_probs, target[:, None].astype(mx.int32), axis=1
                ))

            # Uncertainty weighting
            log_var = self.log_vars[task_name]
            precision = mx.exp(-log_var)
            weighted_loss = precision * task_loss + log_var

            total_loss = total_loss + weighted_loss

        return total_loss


class MultiTaskAccuracy(Trace):
    """Track accuracy for each task."""

    def __init__(self, tasks: dict) -> None:
        """Initialize multi-task accuracy tracker.

        Args:
            tasks: Dictionary mapping task names to (pred_key, target_key).
        """
        self.tasks = tasks
        self.correct = {name: 0 for name in tasks}
        self.total = {name: 0 for name in tasks}

    def on_epoch_begin(self, state):
        self.correct = {name: 0 for name in self.tasks}
        self.total = {name: 0 for name in self.tasks}

    def on_batch_end(self, batch, state):
        for task_name, (pred_key, target_key) in self.tasks.items():
            pred = batch[pred_key]
            target = batch[target_key]

            pred_class = mx.argmax(pred, axis=-1)
            if target.ndim > 1:
                target = mx.argmax(target, axis=-1)

            self.correct[task_name] += int(mx.sum(pred_class == target).item())
            self.total[task_name] += target.shape[0]

    def on_epoch_end(self, state):
        for task_name in self.tasks:
            acc = self.correct[task_name] / max(1, self.total[task_name])
            state['metrics'][f'{task_name}_accuracy'] = acc


def create_multitask_mnist():
    """Create MNIST dataset with multiple tasks.

    Tasks:
    - digit: Classify digit (0-9)
    - odd_even: Classify as odd (1) or even (0)
    """
    from fastmlx.dataset.data import mnist

    train_ds, eval_ds = mnist.load_data()

    # Get data
    train_x = train_ds.data["x"]
    train_y_digit = train_ds.data["y"]
    eval_x = eval_ds.data["x"]
    eval_y_digit = eval_ds.data["y"]

    # Create odd/even labels
    train_y_odd_even = (train_y_digit % 2).astype(mx.int32)
    eval_y_odd_even = (eval_y_digit % 2).astype(mx.int32)

    train_data = MLXDataset({
        "x": train_x,
        "y_digit": train_y_digit,
        "y_odd_even": train_y_odd_even,
    })
    eval_data = MLXDataset({
        "x": eval_x,
        "y_digit": eval_y_digit,
        "y_odd_even": eval_y_odd_even,
    })

    return train_data, eval_data


def get_estimator(
    epochs: int = 20,
    batch_size: int = 64,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create multi-task learning estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        save_dir: Directory to save models.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = create_multitask_mnist()

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[Minmax(inputs="x", outputs="x")],
    )

    # Build multi-task model
    model = fe.build(
        model_fn=lambda: MultiTaskModel(
            input_shape=(1, 28, 28),
            task_outputs={"digit": 10, "odd_even": 2}
        ),
        optimizer_fn="adam"
    )

    network = fe.Network([
        MultiTaskModelOp(
            model=model,
            inputs="x",
            outputs=["pred_digit", "pred_odd_even"]
        ),
        UncertaintyWeightedLoss(
            inputs=["pred_digit", "y_digit", "pred_odd_even", "y_odd_even"],
            outputs="mtl_loss",
            task_names=["digit", "odd_even"],
            num_classes={"digit": 10, "odd_even": 2}
        ),
        fe.op.UpdateOp(model=model, loss_name="mtl_loss")
    ])

    steps_per_epoch = 60000 // batch_size
    cycle_length = epochs * steps_per_epoch

    traces = [
        MultiTaskAccuracy({
            "digit": ("pred_digit", "y_digit"),
            "odd_even": ("pred_odd_even", "y_odd_even"),
        }),
        ModelSaver(model=model, save_dir=save_dir, frequency=5),
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
    parser = argparse.ArgumentParser(description="Multi-Task Learning with FastMLX")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    print("Multi-Task Learning on MNIST")
    print("  Task 1: Digit classification (0-9)")
    print("  Task 2: Odd/Even classification")
    print("  Using uncertainty-weighted loss")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    est.fit()
    est.test()
