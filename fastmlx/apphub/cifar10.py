"""Example CIFAR-10 training script using :mod:`fastmlx`."""

from __future__ import annotations

import tempfile
import fastmlx as fe
from fastmlx.architecture import ResNet9
from fastmlx.dataset.data import cifar10
from fastmlx.op.numpyop.univariate import Minmax
from fastmlx.op.tensorop.loss import CrossEntropy
from fastmlx.op.tensorop.model import ModelOp, UpdateOp
from fastmlx.schedule import cosine_decay
from fastmlx.trace.metric import Accuracy
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.adapt import LRScheduler


def get_estimator(epochs: int = 2, batch_size: int = 64, save_dir: str = tempfile.mkdtemp()) -> fe.Estimator:
    train_data, eval_data = cifar10.load_data()
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           batch_size=batch_size,
                           ops=[Minmax(inputs="x", outputs="x")])
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
