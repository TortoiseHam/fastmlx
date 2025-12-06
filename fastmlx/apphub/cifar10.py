"""Example CIFAR-10 training script using :mod:`fastmlx`."""

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
from fastmlx.trace.metric import Accuracy
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.adapt import LRScheduler

def lr_schedule(step):
    if step <= 490:
        lr = step / 490 * 0.4
    else:
        lr = (2352 - step) / 1862 * 0.4
    return lr * 0.1

def get_estimator(
    epochs: int = 24,
    batch_size: int = 512,
    save_dir: str = tempfile.mkdtemp(),
    num_process: int | None = None,
    mixed_precision: bool = True,
) -> fe.Estimator:
    """Create CIFAR-10 training estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        save_dir: Directory to save best model.
        num_process: Number of data loading processes.
        mixed_precision: Enable mixed precision training (float16).
                        Reduces memory usage and may improve speed.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = cifair10.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(inputs="x", outputs="x", min_height=40, min_width=40),
            RandomCrop(inputs="x", outputs="x", height=32, width=32),
            Sometimes(HorizontalFlip(inputs="x", outputs="x")),
            CoarseDropout(inputs="x", outputs="x", max_holes=1),
            Onehot(inputs="y", outputs="y", num_classes=10, label_smoothing=0.2),
        ],
        num_process=num_process,
    )
    model = fe.build(model_fn=lambda: ResNet9(input_shape=(3, 32, 32)), optimizer_fn="adam")
    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy"),
        LRScheduler(model=model, lr_fn=lr_schedule)
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
    parser = argparse.ArgumentParser(description="CIFAR-10 training with FastMLX")
    parser.add_argument("--epochs", type=int, default=24, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--mixed-precision", "--amp", action="store_true",
                        help="Enable mixed precision training (float16)")
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
