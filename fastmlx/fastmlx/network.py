from typing import List
from .op.op import Op


class Network:
    def __init__(self, ops: List[Op]):
        self.ops = ops

    def run(self, batch, state):
        store = batch
        for op in self.ops:
            inp = store[op.inputs[0]] if len(op.inputs)==1 else [store[k] for k in op.inputs]
            out = op.forward(inp, state)
            if op.outputs:
                if len(op.outputs)==1:
                    store[op.outputs[0]] = out
                else:
                    for k,v in zip(op.outputs,out):
                        store[k]=v
        return store

    def get_loss_keys(self):
        keys = set()
        for op in self.ops:
            if isinstance(op, Op) and op.outputs:
                for k in op.outputs:
                    if "loss" in k:
                        keys.add(k)
        return keys
