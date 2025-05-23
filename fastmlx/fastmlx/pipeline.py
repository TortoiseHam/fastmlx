from typing import Iterable, List


class Pipeline:
    def __init__(self, train_data, eval_data=None, batch_size=32, ops: List=None):
        self.train_data = train_data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.ops = ops or []

    def _loader(self, dataset):
        for i in range(0, len(dataset), self.batch_size):
            batch = {k: v[i:i+self.batch_size] for k, v in dataset.data.items()}
            state = {}
            for op in self.ops:
                inp = batch[op.inputs[0]] if len(op.inputs)==1 else [batch[k] for k in op.inputs]
                out = op.forward(inp, state)
                if op.outputs:
                    if len(op.outputs)==1:
                        batch[op.outputs[0]] = out
                    else:
                        for k,v in zip(op.outputs,out):
                            batch[k]=v
            yield batch

    def get_loader(self, mode="train"):
        dataset = self.train_data if mode=="train" else self.eval_data
        return self._loader(dataset)
