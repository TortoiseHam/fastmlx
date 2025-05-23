"""A simple sequential network abstraction."""

from __future__ import annotations

from typing import Iterable, List, MutableMapping, Sequence

from .op.op import Op


class Network:
    """Execute a sequence of :class:`~fastmlx.op.Op` objects."""

    def __init__(self, ops: Sequence[Op]):
        self.ops: List[Op] = list(ops)

    def run(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> MutableMapping[str, object]:
        store = batch
        for op in self.ops:
            inp = store[op.inputs[0]] if len(op.inputs) == 1 else [store[k] for k in op.inputs]
            out = op.forward(inp, state)
            if op.outputs:
                if len(op.outputs) == 1:
                    store[op.outputs[0]] = out
                else:
                    for k, v in zip(op.outputs, out):
                        store[k] = v
        return store

    def get_loss_keys(self) -> set[str]:
        keys: set[str] = set()
        for op in self.ops:
            if isinstance(op, Op) and op.outputs:
                for k in op.outputs:
                    if "loss" in k or k in {"ce"}:
                        keys.add(k)
        return keys
