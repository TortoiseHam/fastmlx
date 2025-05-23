"""Data loading and preprocessing pipeline."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from typing import Iterable, Iterator, List, MutableMapping, Optional

import mlx.core as mx

_dataset = None
_ops: List | None = None


def _init_pool(dataset: Iterable, ops: List) -> None:
    global _dataset, _ops
    _dataset = dataset
    _ops = ops


def _process_index(idx: int) -> MutableMapping[str, mx.array]:
    assert _dataset is not None and _ops is not None
    sample: MutableMapping[str, mx.array] = {k: v[idx] for k, v in _dataset.data.items()}
    state: MutableMapping[str, object] = {}
    for op in _ops:
        inp = sample[op.inputs[0]] if len(op.inputs) == 1 else [sample[k] for k in op.inputs]
        out = op.forward(inp, state)
        if op.outputs:
            if len(op.outputs) == 1:
                sample[op.outputs[0]] = out
            else:
                for k, v in zip(op.outputs, out):
                    sample[k] = v
    return sample


def _collate(samples: Iterable[MutableMapping[str, mx.array]]) -> MutableMapping[str, mx.array]:
    batch_dict: MutableMapping[str, List[mx.array]] = {}
    for sample in samples:
        for k, v in sample.items():
            batch_dict.setdefault(k, []).append(v)
    return {k: mx.stack(v_list) for k, v_list in batch_dict.items()}


class Pipeline:
    """Hold datasets and preprocessing operations."""

    def __init__(self,
                 train_data: Iterable,
                 eval_data: Optional[Iterable] = None,
                 batch_size: int = 32,
                 ops: Optional[Iterable] = None,
                 num_process: Optional[int] = None) -> None:
        self.train_data: Iterable = train_data
        self.eval_data: Optional[Iterable] = eval_data
        self.batch_size: int = batch_size
        self.ops: List = list(ops or [])
        self.num_process: int = mp.cpu_count() if num_process is None else num_process

    def _loader(self, dataset: Iterable) -> Iterator[MutableMapping[str, object]]:
        if self.num_process <= 0:
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
        else:
            with ThreadPool(self.num_process, initializer=_init_pool, initargs=(dataset, self.ops)) as pool:
                for start in range(0, len(dataset), self.batch_size):
                    indices = list(range(start, min(start + self.batch_size, len(dataset))))
                    samples = pool.map(_process_index, indices)
                    yield _collate(samples)

    def get_loader(self, mode: str = "train") -> Iterator[MutableMapping[str, object]]:
        dataset = self.train_data if mode == "train" else self.eval_data
        if dataset is None:
            raise ValueError(f"No dataset available for mode '{mode}'")
        return self._loader(dataset)
