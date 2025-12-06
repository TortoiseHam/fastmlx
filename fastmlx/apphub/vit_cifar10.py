"""Vision Transformer (ViT) on CIFAR-10 using :mod:`fastmlx`.

This example demonstrates image classification using Vision Transformer
on the CIFAR-10 dataset. Uses a smaller ViT configuration suitable for
32x32 images.

Reference:
    Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale", ICLR 2021.
"""

from __future__ import annotations

import argparse
import tempfile

import fastmlx as fe
from fastmlx.architecture import VisionTransformer
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


def ViT_CIFAR(num_classes: int = 10) -> VisionTransformer:
    """Create a ViT configuration suitable for CIFAR-10 (32x32 images).

    Uses smaller patch size (4x4) and reduced model dimensions
    compared to standard ViT configurations.

    Args:
        num_classes: Number of output classes.

    Returns:
        VisionTransformer configured for 32x32 images.
    """
    return VisionTransformer(
        image_size=32,
        patch_size=4,  # 4x4 patches -> 8x8 = 64 patches
        num_classes=num_classes,
        dims=256,
        depth=6,
        num_heads=4,
        mlp_dims=512,
        dropout=0.1,
        channels=3
    )


def get_estimator(
    epochs: int = 100,
    batch_size: int = 128,
    save_dir: str = tempfile.mkdtemp(),
    num_process: int | None = None,
    mixed_precision: bool = False,
) -> fe.Estimator:
    """Create ViT CIFAR-10 training estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        save_dir: Directory to save best model.
        num_process: Number of data loading processes.
        mixed_precision: Enable mixed precision training (float16).

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = cifair10.load_data()

    # Data pipeline with standard CIFAR-10 augmentation
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            # Normalize with CIFAR-10 statistics
            Normalize(
                inputs="x", outputs="x",
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2471, 0.2435, 0.2616)
            ),
            # Data augmentation
            PadIfNeeded(inputs="x", outputs="x", min_height=40, min_width=40),
            RandomCrop(inputs="x", outputs="x", height=32, width=32),
            Sometimes(HorizontalFlip(inputs="x", outputs="x")),
            CoarseDropout(inputs="x", outputs="x", max_holes=1),
            # Label smoothing
            Onehot(inputs="y", outputs="y", num_classes=10, label_smoothing=0.1),
        ],
        num_process=num_process,
    )

    # Build ViT model
    model = fe.build(
        model_fn=lambda: ViT_CIFAR(num_classes=10),
        optimizer_fn="adam"
    )

    # Network
    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # Calculate learning rate schedule
    steps_per_epoch = 50000 // batch_size  # CIFAR-10 has 50k training samples
    total_steps = epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch  # 5 epochs warmup

    # Traces
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

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces,
        mixed_precision=mixed_precision,
    )
    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT CIFAR-10 with FastMLX")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--mixed-precision", "--amp", action="store_true",
                        help="Enable mixed precision training")
    args = parser.parse_args()

    if args.mixed_precision:
        print("Mixed precision training enabled (float16)")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        mixed_precision=args.mixed_precision,
    )
    est.fit()
    est.test()
