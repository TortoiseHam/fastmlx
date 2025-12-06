"""Example MNIST training script using :mod:`fastmlx`."""

from __future__ import annotations

import tempfile

import fastmlx as fe
from fastmlx.architecture import LeNet
from fastmlx.dataset.data import mnist
from fastmlx.op import CrossEntropy, Minmax, ModelOp, UpdateOp
from fastmlx.schedule import cosine_decay
from fastmlx.trace.adapt import LRScheduler
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.metric import Accuracy


def get_estimator(
    epochs: int = 2,
    batch_size: int = 32,
    save_dir: str = tempfile.mkdtemp(),
    num_process: int | None = None,
) -> fe.Estimator:
    train_data, eval_data = mnist.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[Minmax(inputs="x", outputs="x")],
        num_process=num_process,
    )
    model = fe.build(model_fn=lambda: LeNet(input_shape=(1,28,28)), optimizer_fn="adam")
    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy"),
        LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3))
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
