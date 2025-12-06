"""FGSM Adversarial Training example using :mod:`fastmlx`.

Demonstrates adversarial training using the Fast Gradient Sign Method (FGSM)
to improve model robustness against adversarial attacks.

The training alternates between:
1. Standard training on clean examples
2. Adversarial training on FGSM-perturbed examples

Reference:
    Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Any, MutableMapping

import mlx.core as mx
import mlx.nn as nn

import fastmlx as fe
from fastmlx.architecture import LeNet
from fastmlx.dataset.data import mnist
from fastmlx.op import Minmax, CrossEntropy, ModelOp, UpdateOp, Op
from fastmlx.schedule import cosine_decay
from fastmlx.trace.metric import Accuracy
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.adapt import LRScheduler


class FGSMAttack(Op):
    """Generate FGSM adversarial examples.

    Creates adversarial perturbations by taking the sign of the gradient
    of the loss with respect to the input.

    x_adv = x + epsilon * sign(grad_x(loss))

    Args:
        model: The model to attack.
        inputs: Tuple of (image_key, label_key).
        outputs: Key for adversarial examples.
        epsilon: Maximum perturbation magnitude.
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: tuple,
        outputs: str,
        epsilon: float = 0.3
    ) -> None:
        super().__init__(list(inputs), outputs)
        self.model = model
        self.epsilon = epsilon

    def forward(
        self,
        data: tuple,
        state: MutableMapping[str, Any]
    ) -> mx.array:
        x, y = data

        # Function to compute loss for gradient calculation
        def loss_fn(x_input):
            logits = self.model(x_input)
            # Cross-entropy loss
            if y.ndim > 1:
                # One-hot labels
                loss = -mx.sum(y * mx.log(mx.softmax(logits, axis=-1) + 1e-8))
            else:
                # Integer labels
                loss = mx.mean(
                    mx.take_along_axis(
                        -mx.log(mx.softmax(logits, axis=-1) + 1e-8),
                        y[:, None].astype(mx.int32),
                        axis=1
                    )
                )
            return loss

        # Compute gradient of loss with respect to input
        grad_fn = mx.grad(loss_fn)
        grad_x = grad_fn(x)

        # FGSM perturbation: epsilon * sign(gradient)
        perturbation = self.epsilon * mx.sign(grad_x)

        # Create adversarial example and clip to valid range [0, 1]
        x_adv = mx.clip(x + perturbation, 0.0, 1.0)

        return x_adv


class AdversarialAccuracy(Accuracy):
    """Track accuracy on adversarial examples."""

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred_adv",
        output_name: str = "adv_accuracy"
    ) -> None:
        super().__init__(true_key, pred_key, output_name)


class AdversarialTrainingOp(Op):
    """Combined clean + adversarial training operation.

    Computes loss on both clean and adversarial examples.
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: tuple,
        outputs: str,
        epsilon: float = 0.3,
        adv_weight: float = 0.5
    ) -> None:
        """Initialize adversarial training op.

        Args:
            model: The model to train.
            inputs: Tuple of (pred_clean, pred_adv, label) keys.
            outputs: Key for combined loss.
            epsilon: FGSM epsilon (for reference).
            adv_weight: Weight for adversarial loss (0-1).
        """
        super().__init__(list(inputs), outputs)
        self.model = model
        self.epsilon = epsilon
        self.adv_weight = adv_weight

    def forward(
        self,
        data: tuple,
        state: MutableMapping[str, Any]
    ) -> mx.array:
        y_pred, y_pred_adv, y = data

        # Clean loss
        if y.ndim > 1:
            clean_loss = -mx.mean(mx.sum(y * mx.log(mx.softmax(y_pred, axis=-1) + 1e-8), axis=-1))
            adv_loss = -mx.mean(mx.sum(y * mx.log(mx.softmax(y_pred_adv, axis=-1) + 1e-8), axis=-1))
        else:
            log_probs = mx.log(mx.softmax(y_pred, axis=-1) + 1e-8)
            clean_loss = mx.mean(
                mx.take_along_axis(-log_probs, y[:, None].astype(mx.int32), axis=1)
            )
            log_probs_adv = mx.log(mx.softmax(y_pred_adv, axis=-1) + 1e-8)
            adv_loss = mx.mean(
                mx.take_along_axis(-log_probs_adv, y[:, None].astype(mx.int32), axis=1)
            )

        # Combined loss
        total_loss = (1 - self.adv_weight) * clean_loss + self.adv_weight * adv_loss

        return total_loss


def get_estimator(
    epochs: int = 10,
    batch_size: int = 64,
    epsilon: float = 0.3,
    adv_weight: float = 0.5,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create FGSM adversarial training estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        epsilon: FGSM perturbation magnitude (0-1).
        adv_weight: Weight for adversarial loss in combined training.
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

    # Network with adversarial training
    network = fe.Network([
        # Forward pass on clean examples
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        # Generate adversarial examples
        FGSMAttack(model=model, inputs=("x", "y"), outputs="x_adv", epsilon=epsilon),
        # Forward pass on adversarial examples
        ModelOp(model=model, inputs="x_adv", outputs="y_pred_adv"),
        # Combined loss
        AdversarialTrainingOp(
            model=model,
            inputs=("y_pred", "y_pred_adv", "y"),
            outputs="adv_loss",
            epsilon=epsilon,
            adv_weight=adv_weight
        ),
        UpdateOp(model=model, loss_name="adv_loss")
    ])

    steps_per_epoch = 60000 // batch_size
    cycle_length = epochs * steps_per_epoch

    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="clean_accuracy"),
        AdversarialAccuracy(true_key="y", pred_key="y_pred_adv"),
        BestModelSaver(model=model, save_dir=save_dir, metric="adv_accuracy"),
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

    return fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FGSM Adversarial Training with FastMLX"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="FGSM perturbation magnitude (0-1)")
    parser.add_argument("--adv-weight", type=float, default=0.5,
                        help="Weight for adversarial loss (0-1)")
    args = parser.parse_args()

    print("FGSM Adversarial Training")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Adversarial weight: {args.adv_weight}")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        epsilon=args.epsilon,
        adv_weight=args.adv_weight,
    )
    est.fit()
    est.test()
