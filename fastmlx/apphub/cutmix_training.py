"""CutMix training example using :mod:`fastmlx`.

Demonstrates CutMix data augmentation which cuts and pastes patches
between training images while mixing their labels proportionally.

CutMix addresses the shortcomings of Cutout (information loss) and
MixUp (unnatural blending) by creating locally natural training samples.

Reference:
    Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers
    with Localizable Features", ICCV 2019.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Tuple

import mlx.core as mx
import numpy as np

import fastmlx as fe
from fastmlx.architecture import ResNet9
from fastmlx.dataset.data import cifair10
from fastmlx.op import (
    CrossEntropy,
    HorizontalFlip,
    ModelOp,
    Normalize,
    Onehot,
    Op,
    PadIfNeeded,
    RandomCrop,
    Sometimes,
    UpdateOp,
)
from fastmlx.schedule import warmup_cosine_decay
from fastmlx.trace.adapt import LRScheduler
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.metric import Accuracy


class CutMix(Op):
    """CutMix augmentation operation.

    Cuts a random patch from one image and pastes it onto another,
    mixing labels according to the patch area.

    Args:
        inputs: Tuple of (images, labels) keys.
        outputs: Tuple of (mixed_images, mixed_labels) keys.
        alpha: Parameter for Beta distribution to sample mixing ratio.
        prob: Probability of applying CutMix.
    """

    def __init__(
        self,
        inputs: Tuple[str, str],
        outputs: Tuple[str, str],
        alpha: float = 1.0,
        prob: float = 0.5
    ) -> None:
        super().__init__(list(inputs), list(outputs))
        self.alpha = alpha
        self.prob = prob

    def forward(self, data, state):
        images, labels = data

        if state.get("mode") != "train" or np.random.random() > self.prob:
            return images, labels

        batch_size = images.shape[0]
        height, width = images.shape[1], images.shape[2]

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Random permutation for mixing
        indices = mx.random.permutation(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]

        # Calculate cut box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)

        # Random position for the cut
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)

        x1 = max(0, cx - cut_w // 2)
        x2 = min(width, cx + cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(height, cy + cut_h // 2)

        # Calculate actual lambda based on cut area
        actual_lam = 1 - (x2 - x1) * (y2 - y1) / (height * width)

        # Create mixed images
        mixed_images = images.astype(mx.float32)
        # Replace the cut region with shuffled image
        patch = shuffled_images[:, y1:y2, x1:x2, :]
        mixed_images = mixed_images.at[:, y1:y2, x1:x2, :].set(patch)

        # Mix labels
        mixed_labels = actual_lam * labels + (1 - actual_lam) * shuffled_labels

        return mixed_images, mixed_labels


def get_estimator(
    epochs: int = 100,
    batch_size: int = 128,
    alpha: float = 1.0,
    cutmix_prob: float = 0.5,
    save_dir: str = tempfile.mkdtemp(),
    num_process: int | None = None,
    mixed_precision: bool = False,
) -> fe.Estimator:
    """Create CutMix training estimator for CIFAR-10.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        alpha: CutMix alpha parameter for Beta distribution.
        cutmix_prob: Probability of applying CutMix to a batch.
        save_dir: Directory to save best model.
        num_process: Number of data loading processes.
        mixed_precision: Enable mixed precision training.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = cifair10.load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x",
                     mean=(0.4914, 0.4822, 0.4465),
                     std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(inputs="x", outputs="x", min_height=40, min_width=40),
            RandomCrop(inputs="x", outputs="x", height=32, width=32),
            Sometimes(HorizontalFlip(inputs="x", outputs="x")),
            # Convert to one-hot before CutMix
            Onehot(inputs="y", outputs="y", num_classes=10),
            # Apply CutMix
            CutMix(inputs=("x", "y"), outputs=("x", "y"),
                  alpha=alpha, prob=cutmix_prob),
        ],
        num_process=num_process,
    )

    model = fe.build(
        model_fn=lambda: ResNet9(input_shape=(3, 32, 32)),
        optimizer_fn="adam"
    )

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    steps_per_epoch = 50000 // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy"),
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
        traces=traces,
        mixed_precision=mixed_precision,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CutMix Training with FastMLX")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="CutMix alpha (Beta distribution parameter)")
    parser.add_argument("--cutmix-prob", type=float, default=0.5,
                        help="Probability of applying CutMix")
    parser.add_argument("--mixed-precision", "--amp", action="store_true",
                        help="Enable mixed precision training")
    args = parser.parse_args()

    print("CutMix Training on CIFAR-10")
    print(f"  Alpha: {args.alpha}")
    print(f"  CutMix probability: {args.cutmix_prob}")
    if args.mixed_precision:
        print("  Mixed precision: enabled")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        cutmix_prob=args.cutmix_prob,
        mixed_precision=args.mixed_precision,
    )
    est.fit()
    est.test()
