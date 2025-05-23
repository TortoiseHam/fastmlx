"""Data loading and preprocessing pipeline."""

from __future__ import annotations

from typing import Iterable, Iterable, Iterator, List, MutableMapping, Optional


class Pipeline:
    """Hold datasets and preprocessing operations."""

    def __init__(self, train_data: Iterable, eval_data: Optional[Iterable] = None,
                 batch_size: int = 32, ops: Optional[Iterable] = None) -> None:
        self.train_data: Iterable = train_data
        self.eval_data: Optional[Iterable] = eval_data
        self.batch_size: int = batch_size
        self.ops: List = list(ops or [])

    def _loader(self, dataset: Iterable) -> Iterator[MutableMapping[str, object]]:
        for i in range(0, len(dataset), self.batch_size):
            batch: MutableMapping[str, object] = {k: v[i:i + self.batch_size] for k, v in dataset.data.items()}
            state: MutableMapping[str, object] = {}
            for op in self.ops:
                inp = batch[op.inputs[0]] if len(op.inputs) == 1 else [batch[k] for k in op.inputs]
                out = op.forward(inp, state)
                if op.outputs:
                    if len(op.outputs) == 1:
                        batch[op.outputs[0]] = out
                    else:
                        for k, v in zip(op.outputs, out):
                            batch[k] = v
            yield batch

    def get_loader(self, mode: str = "train") -> Iterator[MutableMapping[str, object]]:
        dataset = self.train_data if mode == "train" else self.eval_data
        if dataset is None:
            raise ValueError(f"No dataset available for mode '{mode}'")
        return self._loader(dataset)
