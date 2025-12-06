"""Transfer Learning / Fine-tuning example using :mod:`fastmlx`.

Demonstrates fine-tuning a pre-trained model on a new dataset.
The approach:
1. Load a pre-trained model (trained on one dataset)
2. Replace/modify the classification head for new task
3. Fine-tune with lower learning rate (optionally freeze early layers)

This is commonly used when:
- You have limited data for your target task
- You want to leverage features learned on a larger dataset
- You need faster convergence

This example shows fine-tuning a CIFAR-10 model on Fashion-MNIST
(demonstrating domain transfer, though in practice you'd use
more similar domains).
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

import fastmlx as fe
from fastmlx.architecture import LeNet
from fastmlx.dataset.data import mnist, fashion_mnist
from fastmlx.op import Minmax, CrossEntropy, ModelOp, UpdateOp
from fastmlx.schedule import warmup_cosine_decay
from fastmlx.trace.metric import Accuracy
from fastmlx.trace.io import BestModelSaver, ModelSaver
from fastmlx.trace.adapt import LRScheduler


class TransferLeNet(nn.Module):
    """LeNet with replaceable classification head for transfer learning."""

    def __init__(
        self,
        input_shape: tuple = (1, 28, 28),
        num_classes: int = 10,
        freeze_features: bool = False
    ) -> None:
        super().__init__()
        self.freeze_features = freeze_features

        in_channels = input_shape[0]

        # Feature extractor (can be frozen)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification head (always trainable)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.freeze_features:
            # Detach features from gradient computation
            with mx.no_grad():
                features = self.features(x)
            features = mx.stop_gradient(features)
        else:
            features = self.features(x)

        features = features.reshape(features.shape[0], -1)
        return self.classifier(features)

    def get_feature_params(self):
        """Get parameters from feature extractor."""
        return self.features.parameters()

    def get_classifier_params(self):
        """Get parameters from classifier head."""
        return self.classifier.parameters()


def pretrain_on_mnist(
    epochs: int = 5,
    batch_size: int = 64,
    save_path: str = None
) -> TransferLeNet:
    """Pre-train model on MNIST.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        save_path: Path to save pre-trained weights.

    Returns:
        Pre-trained model.
    """
    print("Pre-training on MNIST...")

    train_data, eval_data = mnist.load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[Minmax(inputs="x", outputs="x")],
    )

    model = fe.build(
        model_fn=lambda: TransferLeNet(input_shape=(1, 28, 28), num_classes=10),
        optimizer_fn="adam"
    )

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
    ]

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )
    estimator.fit()

    if save_path:
        model.save_weights(save_path)
        print(f"Pre-trained model saved to {save_path}")

    return model


def finetune_on_fashion_mnist(
    pretrained_model: TransferLeNet,
    epochs: int = 10,
    batch_size: int = 64,
    freeze_features: bool = True,
    lr_features: float = 1e-4,
    lr_classifier: float = 1e-3,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Fine-tune pre-trained model on Fashion-MNIST.

    Args:
        pretrained_model: Pre-trained model from MNIST.
        epochs: Number of fine-tuning epochs.
        batch_size: Batch size.
        freeze_features: Whether to freeze feature extractor.
        lr_features: Learning rate for feature extractor (if not frozen).
        lr_classifier: Learning rate for classifier head.
        save_dir: Directory to save fine-tuned model.

    Returns:
        Configured Estimator ready for training.
    """
    print(f"\nFine-tuning on Fashion-MNIST...")
    print(f"  Freeze features: {freeze_features}")
    if not freeze_features:
        print(f"  Feature LR: {lr_features}, Classifier LR: {lr_classifier}")

    train_data, eval_data = fashion_mnist.load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[Minmax(inputs="x", outputs="x")],
    )

    # Create new model and copy pre-trained weights
    model = TransferLeNet(
        input_shape=(1, 28, 28),
        num_classes=10,
        freeze_features=freeze_features
    )

    # Copy feature weights from pre-trained model
    model.features.update(pretrained_model.features.parameters())

    # Re-initialize classifier for new task (optional, can also keep and fine-tune)
    # Here we keep the pre-trained classifier structure but it will be fine-tuned
    mx.eval(model.parameters())

    # Build with optimizer
    model = fe.build(
        model_fn=lambda: model,
        optimizer_fn=lambda: nn.optimizers.Adam(learning_rate=lr_classifier)
    )

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    steps_per_epoch = 60000 // batch_size
    total_steps = epochs * steps_per_epoch

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy"),
        LRScheduler(
            model=model,
            lr_fn=lambda step: warmup_cosine_decay(
                step, warmup_steps=steps_per_epoch,
                total_steps=total_steps, init_lr=lr_classifier, min_lr=1e-5
            )
        )
    ]

    return fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )


def train_from_scratch(
    epochs: int = 10,
    batch_size: int = 64,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Train on Fashion-MNIST from scratch for comparison.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        save_dir: Directory to save model.

    Returns:
        Configured Estimator ready for training.
    """
    print("\nTraining from scratch on Fashion-MNIST...")

    train_data, eval_data = fashion_mnist.load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[Minmax(inputs="x", outputs="x")],
    )

    model = fe.build(
        model_fn=lambda: TransferLeNet(input_shape=(1, 28, 28), num_classes=10),
        optimizer_fn="adam"
    )

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    steps_per_epoch = 60000 // batch_size
    total_steps = epochs * steps_per_epoch

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy"),
        LRScheduler(
            model=model,
            lr_fn=lambda step: warmup_cosine_decay(
                step, warmup_steps=steps_per_epoch,
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
    parser = argparse.ArgumentParser(description="Transfer Learning with FastMLX")
    parser.add_argument("--pretrain-epochs", type=int, default=5,
                        help="Epochs for pre-training on MNIST")
    parser.add_argument("--finetune-epochs", type=int, default=10,
                        help="Epochs for fine-tuning on Fashion-MNIST")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--freeze-features", action="store_true",
                        help="Freeze feature extractor during fine-tuning")
    parser.add_argument("--compare-scratch", action="store_true",
                        help="Also train from scratch for comparison")
    args = parser.parse_args()

    print("Transfer Learning: MNIST -> Fashion-MNIST")
    print("=" * 50)

    # Pre-train on MNIST
    pretrained = pretrain_on_mnist(
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
    )

    # Fine-tune on Fashion-MNIST
    finetune_est = finetune_on_fashion_mnist(
        pretrained_model=pretrained,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        freeze_features=args.freeze_features,
    )
    finetune_est.fit()
    finetune_est.test()

    # Optionally compare with training from scratch
    if args.compare_scratch:
        print("\n" + "=" * 50)
        scratch_est = train_from_scratch(
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
        )
        scratch_est.fit()
        scratch_est.test()
