import mlx.nn as nn
import mlx.core as mx
from ..op import Op


class CrossEntropy(Op):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs)

    def forward(self, data, state):
        y_pred, y_true = data
        loss = nn.losses.cross_entropy(y_pred, y_true)
        return mx.mean(loss)
