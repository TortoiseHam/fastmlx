"""WideResNet on CIFAR-10 using :mod:`fastmlx`.

Demonstrates training Wide Residual Networks which achieve better
accuracy than standard ResNets by increasing width instead of depth.

Reference:
    Zagoruyko & Komodakis, "Wide Residual Networks", BMVC 2016.
"""

from __future__ import annotations

import argparse
import tempfile

import fastmlx as fe
from fastmlx.architecture import WideResNet16_8, WideResNet28_10, WideResNet40_4
from fastmlx.dataset.data import cifair10
from fastmlx.op import (
    CoarseDropout,
    CrossEntropy,
    HorizontalFlip,
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

ARCHITECTURES = {
    "wrn-16-8": WideResNet16_8,
    "wrn-28-10": WideResNet28_10,
    "wrn-40-4": WideResNet40_4,
}


def get_estimator(
    epochs: int = 200,
    batch_size: int = 128,
    architecture: str = "wrn-28-10",
    save_dir: str = tempfile.mkdtemp(),
    num_process: int | None = None,
    mixed_precision: bool = False,
) -> fe.Estimator:
    """Create WideResNet CIFAR-10 training estimator.

    Args:
        epochs: Number of training epochs. WideResNets typically need
                more epochs for best results (200+).
        batch_size: Batch size for training.
        architecture: Which WideResNet variant to use:
                     - 'wrn-16-8': 16 layers, width 8 (fastest)
                     - 'wrn-28-10': 28 layers, width 10 (best accuracy)
                     - 'wrn-40-4': 40 layers, width 4
        save_dir: Directory to save best model.
        num_process: Number of data loading processes.
        mixed_precision: Enable mixed precision training.

    Returns:
        Configured Estimator ready for training.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Choose from: {list(ARCHITECTURES.keys())}")

    train_data, eval_data = cifair10.load_data()

    # Standard CIFAR-10 preprocessing with strong augmentation
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
            Onehot(inputs="y", outputs="y", num_classes=10, label_smoothing=0.1),
        ],
        num_process=num_process,
    )

    # Build WideResNet model
    model_class = ARCHITECTURES[architecture]
    model = fe.build(
        model_fn=lambda: model_class(classes=10, input_shape=(3, 32, 32)),
        optimizer_fn="adam"
    )

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # Learning rate schedule with warmup
    steps_per_epoch = 50000 // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch

    traces = [
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
    parser = argparse.ArgumentParser(description="WideResNet CIFAR-10 with FastMLX")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--architecture", type=str, default="wrn-28-10",
                        choices=list(ARCHITECTURES.keys()),
                        help="WideResNet variant")
    parser.add_argument("--mixed-precision", "--amp", action="store_true",
                        help="Enable mixed precision training")
    args = parser.parse_args()

    print(f"Training {args.architecture.upper()} on CIFAR-10")
    if args.mixed_precision:
        print("Mixed precision training enabled")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        architecture=args.architecture,
        mixed_precision=args.mixed_precision,
    )
    est.fit()
    est.test()
