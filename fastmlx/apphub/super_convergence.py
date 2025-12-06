"""Super Convergence training example using :mod:`fastmlx`.

Demonstrates the 1cycle learning rate policy for faster training
convergence on CIFAR-10 with ResNet9.

The key insight is to use large learning rates with a specific schedule:
- Warmup phase: LR increases from min to max
- Annealing phase: LR decreases from max back to min

This allows training with larger batch sizes and fewer epochs while
achieving competitive accuracy.

Reference:
    Smith & Topin, "Super-Convergence: Very Fast Training of Neural
    Networks Using Large Learning Rates", 2019.
"""

from __future__ import annotations

import argparse
import tempfile

import fastmlx as fe
from fastmlx.architecture import ResNet9
from fastmlx.dataset.data import cifair10
from fastmlx.op import (
    Normalize,
    PadIfNeeded,
    RandomCrop,
    HorizontalFlip,
    CoarseDropout,
    Onehot,
    Sometimes,
    CrossEntropy,
    ModelOp,
    UpdateOp,
)
from fastmlx.schedule import one_cycle
from fastmlx.trace.metric import Accuracy
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.adapt import LRScheduler


def get_estimator(
    epochs: int = 24,
    batch_size: int = 512,
    max_lr: float = 0.4,
    min_lr: float = 0.0,
    pct_start: float = 0.3,
    save_dir: str = tempfile.mkdtemp(),
    num_process: int | None = None,
    mixed_precision: bool = True,
) -> fe.Estimator:
    """Create super convergence estimator for CIFAR-10.

    Args:
        epochs: Number of training epochs. Super convergence allows
                achieving good results with fewer epochs (e.g., 24).
        batch_size: Batch size. Larger batches work well with super
                   convergence (e.g., 512 or 1024).
        max_lr: Maximum learning rate during the cycle. This can be
               much higher than typical (e.g., 0.4).
        min_lr: Minimum learning rate at start and end.
        pct_start: Fraction of training for the warmup phase.
        save_dir: Directory to save best model.
        num_process: Number of data loading processes.
        mixed_precision: Enable mixed precision training.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = cifair10.load_data()

    # Standard CIFAR-10 preprocessing
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
            PadIfNeeded(inputs="x", outputs="x", min_height=40, min_width=40),
            RandomCrop(inputs="x", outputs="x", height=32, width=32),
            Sometimes(HorizontalFlip(inputs="x", outputs="x")),
            CoarseDropout(inputs="x", outputs="x", max_holes=1),
            Onehot(inputs="y", outputs="y", num_classes=10, label_smoothing=0.2),
        ],
        num_process=num_process,
    )

    # Build model
    model = fe.build(
        model_fn=lambda: ResNet9(input_shape=(3, 32, 32)),
        optimizer_fn="adam"
    )

    # Network
    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # Calculate total steps
    steps_per_epoch = 50000 // batch_size
    total_steps = epochs * steps_per_epoch

    # 1cycle learning rate schedule
    def lr_fn(step: int) -> float:
        """One cycle learning rate with momentum scheduling."""
        return one_cycle(
            step=step,
            total_steps=total_steps,
            max_lr=max_lr,
            min_lr=min_lr,
            pct_start=pct_start,
            anneal_strategy="cos"
        )

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy"),
        LRScheduler(model=model, lr_fn=lr_fn)
    ]

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces,
        mixed_precision=mixed_precision,
    )
    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Super Convergence Training with FastMLX"
    )
    parser.add_argument("--epochs", type=int, default=24,
                        help="Number of epochs (default: 24)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--max-lr", type=float, default=0.4,
                        help="Maximum learning rate (default: 0.4)")
    parser.add_argument("--pct-start", type=float, default=0.3,
                        help="Fraction for warmup phase (default: 0.3)")
    parser.add_argument("--mixed-precision", "--amp", action="store_true",
                        help="Enable mixed precision training")
    args = parser.parse_args()

    print("Super Convergence Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max LR: {args.max_lr}")
    print(f"  Warmup fraction: {args.pct_start}")
    if args.mixed_precision:
        print("  Mixed precision: enabled")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_lr=args.max_lr,
        pct_start=args.pct_start,
        mixed_precision=args.mixed_precision,
    )
    est.fit()
    est.test()
