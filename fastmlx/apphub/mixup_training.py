"""MixUp training example using :mod:`fastmlx`.

Demonstrates MixUp data augmentation which creates virtual training
examples by linearly interpolating between pairs of examples and their labels.

MixUp improves generalization and calibration, and makes models more
robust to adversarial examples and label noise.

Reference:
    Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
"""

from __future__ import annotations

import argparse
import tempfile

import fastmlx as fe
from fastmlx.architecture import ResNet9
from fastmlx.dataset.data import cifair10
from fastmlx.op import (
    CrossEntropy,
    HorizontalFlip,
    MixUp,
    ModelOp,
    Normalize,
    Onehot,
    PadIfNeeded,
    RandomCrop,
    Sometimes,
    UpdateOp,
)
from fastmlx.schedule import warmup_cosine_decay
from fastmlx.trace.adapt import LRScheduler
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.metric import Accuracy


def get_estimator(
    epochs: int = 100,
    batch_size: int = 128,
    alpha: float = 1.0,
    save_dir: str = tempfile.mkdtemp(),
    num_process: int | None = None,
    mixed_precision: bool = False,
) -> fe.Estimator:
    """Create MixUp training estimator for CIFAR-10.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        alpha: MixUp interpolation strength. Higher values create more
              mixed examples. Common values: 0.2-1.0.
              - alpha=0: No mixing (standard training)
              - alpha=1: Uniform distribution over mixing ratios
        save_dir: Directory to save best model.
        num_process: Number of data loading processes.
        mixed_precision: Enable mixed precision training.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = cifair10.load_data()

    # Pipeline with MixUp applied after other augmentations
    # Note: MixUp should be applied AFTER converting labels to one-hot
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(
                inputs="x", outputs="x",
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2471, 0.2435, 0.2616)
            ),
            # Standard augmentation
            PadIfNeeded(inputs="x", outputs="x", min_height=40, min_width=40),
            RandomCrop(inputs="x", outputs="x", height=32, width=32),
            Sometimes(HorizontalFlip(inputs="x", outputs="x")),
            # Convert to one-hot before MixUp (required for soft labels)
            Onehot(inputs="y", outputs="y", num_classes=10),
            # Apply MixUp to both images and labels
            MixUp(inputs=("x", "y"), outputs=("x", "y"), alpha=alpha),
        ],
        num_process=num_process,
    )

    model = fe.build(
        model_fn=lambda: ResNet9(input_shape=(3, 32, 32)),
        optimizer_fn="adam"
    )

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        # CrossEntropy with soft labels (from MixUp)
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    steps_per_epoch = 50000 // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch

    traces = [
        # Note: Accuracy will be lower during training due to soft labels
        # but should be evaluated on original labels during eval
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy"),
        LRScheduler(
            model=model,
            lr_fn=lambda step: warmup_cosine_decay(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                init_lr=1e-3,
                min_lr=1e-5
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
    parser = argparse.ArgumentParser(description="MixUp Training with FastMLX")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="MixUp alpha (higher = more mixing)")
    parser.add_argument("--mixed-precision", "--amp", action="store_true",
                        help="Enable mixed precision training")
    args = parser.parse_args()

    print("MixUp Training on CIFAR-10")
    print(f"  Alpha: {args.alpha}")
    if args.mixed_precision:
        print("  Mixed precision: enabled")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        mixed_precision=args.mixed_precision,
    )
    est.fit()
    est.test()
