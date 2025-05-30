"""Data loading and preprocessing pipeline."""

from __future__ import annotations

import multiprocessing as mp
from typing import Iterable, Iterator, List, MutableMapping, Optional

import mlx.core as mx
import mlx.data as dx

def _process_sample(
    sample: MutableMapping[str, mx.array],
    ops: List,
    state: Optional[MutableMapping[str, object]] = None,
) -> MutableMapping[str, mx.array]:
    """Apply Ops to a single sample."""

    if state is None:
        state = {}

    for op in ops:
        inp = sample[op.inputs[0]] if len(op.inputs) == 1 else [sample[k] for k in op.inputs]
        out = op.forward(inp, state)
        if op.outputs:
            if len(op.outputs) == 1:
                sample[op.outputs[0]] = out
            else:
                for k, v in zip(op.outputs, out):
                    sample[k] = v
    return sample


class Pipeline:
    """Hold datasets and preprocessing operations."""

    def __init__(self,
                 train_data: Iterable,
                 eval_data: Optional[Iterable] = None,
                 batch_size: int = 32,
                 ops: Optional[Iterable] = None,
                 num_process: Optional[int] = 0) -> None:
        self.train_data: Iterable = train_data
        self.eval_data: Optional[Iterable] = eval_data
        self.batch_size: int = batch_size
        self.ops: List = list(ops or [])
        self.num_process: int = mp.cpu_count() if num_process is None else num_process

    def _loader(self, dataset: Iterable) -> Iterator[MutableMapping[str, object]]:
        """Create an iterator over processed batches."""

        if isinstance(dataset, dx.Buffer):
            buffer = dataset

            def transform(sample: MutableMapping[str, object]) -> MutableMapping[str, object]:
                mx_sample = {k: mx.array(v) for k, v in sample.items()}
                return _process_sample(mx_sample, self.ops, {})

            stream = buffer.sample_transform(transform)
            stream = stream.batch(self.batch_size)
            if self.num_process and self.num_process > 0:
                stream = stream.ordered_prefetch(self.batch_size * 2, self.num_process)

            for batch in stream:
                # ``stream`` yields numpy arrays, convert to MLX arrays
                yield {k: mx.array(v) for k, v in batch.items()}

        elif hasattr(dataset, "data") and hasattr(dataset, "__len__"):
            arrays = {k: mx.array(v) for k, v in dataset.data.items()}
            size = len(dataset)

            def apply_ops(batch: MutableMapping[str, mx.array]) -> MutableMapping[str, mx.array]:
                state: MutableMapping[str, object] = {}
                for op in self.ops:
                    inp = batch[op.inputs[0]] if len(op.inputs) == 1 else [batch[k] for k in op.inputs]
                    out = op.forward(inp, state)
                    if op.outputs:
                        if len(op.outputs) == 1:
                            batch[op.outputs[0]] = out
                        else:
                            for k_out, v_out in zip(op.outputs, out):
                                batch[k_out] = v_out
                return batch

            for start in range(0, size, self.batch_size):
                batch = {k: v[start:start + self.batch_size] for k, v in arrays.items()}
                batch = apply_ops(batch)
                yield batch

        else:
            buffer = dx.buffer_from_vector(list(dataset))

            def transform(sample: MutableMapping[str, object]) -> MutableMapping[str, object]:
                mx_sample = {k: mx.array(v) for k, v in sample.items()}
                return _process_sample(mx_sample, self.ops, {})

            stream = buffer.sample_transform(transform)
            stream = stream.batch(self.batch_size)
            if self.num_process and self.num_process > 0:
                stream = stream.ordered_prefetch(self.batch_size * 2, self.num_process)

            for batch in stream:
                yield {k: mx.array(v) for k, v in batch.items()}

    def get_loader(self, mode: str = "train") -> Iterator[MutableMapping[str, object]]:
        dataset = self.train_data if mode == "train" else self.eval_data
        if dataset is None:
            raise ValueError(f"No dataset available for mode '{mode}'")
        return self._loader(dataset)
