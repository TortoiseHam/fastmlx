from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np

from .op import Op


class Sometimes(Op):
    """Execute an op probabilistically."""

    def __init__(self, op: Op, prob: float = 0.5) -> None:
        super().__init__(op.inputs, op.outputs)
        self.op = op
        self.prob = prob

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Any:
        if np.random.rand() < self.prob:
            return self.op.forward(data, state)
        return data
