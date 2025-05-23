import mlx.nn as nn
import mlx.core as mx
from ..op import Op


class ModelOp(Op):
    def __init__(self, model, inputs, outputs):
        super().__init__(inputs, outputs)
        self.model = model

    def forward(self, data, state):
        return self.model(data)


class UpdateOp(Op):
    def __init__(self, model, loss_name):
        super().__init__(inputs=loss_name, outputs=None)
        self.model = model

    def forward(self, data, state):
        loss = data
        grads = mx.grad(loss, self.model.trainable_parameters())
        self.model.optimizer.update(self.model, grads)
        return None
