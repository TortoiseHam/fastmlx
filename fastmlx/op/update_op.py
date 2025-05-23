from __future__ import annotations

from typing import Any, MutableMapping, List

import mlx.nn as nn
import mlx.core as mx
from functools import partial

from .op import Op

Array = mx.array


class UpdateOp(Op):
    """Compute gradients and update model parameters."""

    def __init__(self, model: nn.Module, loss_name: str, compile: bool = True) -> None:
        super().__init__(inputs=loss_name, outputs=None)
        self.model: nn.Module = model
        self._state: List[mx.array] = [self.model.state, self.model.optimizer.state, mx.random.state]

        def step(x_data: Array, y_data: Array) -> Array:
            def loss_fn(m: nn.Module, x_in: Array, y_in: Array) -> Array:
                logits = m(x_in)
                loss = nn.losses.cross_entropy(logits, y_in)
                return mx.mean(loss)

            loss_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss_val, grads = loss_grad_fn(self.model, x_data, y_data)
            self.model.optimizer.update(self.model, grads)
            return loss_val

        if compile:
            self._step = partial(mx.compile, inputs=self._state, outputs=self._state)(step)
        else:
            self._step = step

    def forward(self, data: Array, state: MutableMapping[str, Any]) -> None:
        batch = state.get("batch", {})
        x = batch.get("x")
        y = batch.get("y")
        if x is None or y is None:
            return None
        if state.get("mode") != "train":
            return None

        loss_val = self._step(x, y)
        batch[self.inputs[0]] = loss_val
        mx.eval(loss_val, *self._state)
        return None
