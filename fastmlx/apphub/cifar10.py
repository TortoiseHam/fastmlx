"""Example CIFAR-10 training script using :mod:`fastmlx`."""

from __future__ import annotations

import tempfile
import fastmlx as fe
from fastmlx.architecture import ResNet9
from fastmlx.dataset.data import cifar10
from fastmlx.op.numpyop import (
    Normalize,
    PadIfNeeded,
    RandomCrop,
    HorizontalFlip,
    CoarseDropout,
    Onehot,
    Sometimes,
)
from fastmlx.op.tensorop.loss import CrossEntropy
from fastmlx.op.tensorop.model import ModelOp, UpdateOp
from fastmlx.schedule import cosine_decay
from fastmlx.trace.metric import Accuracy
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.adapt import LRScheduler


def get_estimator(epochs: int = 2, batch_size: int = 64, save_dir: str = tempfile.mkdtemp()) -> fe.Estimator:
    train_data, eval_data = cifar10.load_data()
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
        LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=7500, init_lr=1e-3))
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
