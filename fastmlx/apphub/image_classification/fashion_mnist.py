"""Fashion-MNIST classification example using :mod:`fastmlx`.

Fashion-MNIST is a dataset of Zalando's article images, designed as a
drop-in replacement for MNIST with increased difficulty.

Classes:
    0: T-shirt/top
    1: Trouser
    2: Pullover
    3: Dress
    4: Coat
    5: Sandal
    6: Shirt
    7: Sneaker
    8: Bag
    9: Ankle boot

Reference:
    Xiao et al., "Fashion-MNIST: a Novel Image Dataset for Benchmarking
    Machine Learning Algorithms", 2017.
"""

from __future__ import annotations

import argparse
import tempfile

import fastmlx as fe
from fastmlx.architecture import LeNet
from fastmlx.dataset.data import fashion_mnist
from fastmlx.op import (
    CrossEntropy,
    HorizontalFlip,
    Minmax,
    ModelOp,
    PadIfNeeded,
    RandomCrop,
    Sometimes,
    UpdateOp,
)
from fastmlx.schedule import cosine_decay
from fastmlx.trace.adapt import LRScheduler
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.metric import Accuracy


def get_estimator(
    epochs: int = 10,
    batch_size: int = 64,
    save_dir: str = tempfile.mkdtemp(),
    num_process: int | None = None,
) -> fe.Estimator:
    """Create Fashion-MNIST training estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        save_dir: Directory to save best model.
        num_process: Number of data loading processes.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = fashion_mnist.load_data()

    # Pipeline with light augmentation
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Minmax(inputs="x", outputs="x"),
            # Light data augmentation
            PadIfNeeded(inputs="x", outputs="x", min_height=32, min_width=32),
            RandomCrop(inputs="x", outputs="x", height=28, width=28),
            Sometimes(HorizontalFlip(inputs="x", outputs="x"), prob=0.5),
        ],
        num_process=num_process,
    )

    # Build LeNet model
    model = fe.build(
        model_fn=lambda: LeNet(input_shape=(1, 28, 28)),
        optimizer_fn="adam"
    )

    # Network
    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # Calculate schedule parameters
    steps_per_epoch = 60000 // batch_size
    cycle_length = epochs * steps_per_epoch

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy"),
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

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )
    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST Classification with FastMLX"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    print("Fashion-MNIST Classification")
    print("Classes: T-shirt, Trouser, Pullover, Dress, Coat, "
          "Sandal, Shirt, Sneaker, Bag, Ankle boot")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    est.fit()
    est.test()
