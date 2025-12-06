"""Data loading and preprocessing pipeline."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Iterable, Iterator, List, MutableMapping, Optional

import mlx.core as mx
import mlx.data as dx


class PipelineError(Exception):
    """Exception raised for pipeline processing errors."""
    pass


def _validate_op_inputs(
    op: Any,
    sample: MutableMapping[str, Any],
    op_name: str = "Op"
) -> None:
    """Validate that all input keys exist in the sample.

    Args:
        op: The operation to validate.
        sample: The current sample dictionary.
        op_name: Name for error messages.

    Raises:
        PipelineError: If a required input key is missing.
    """
    for key in op.inputs:
        if key not in sample:
            available = list(sample.keys())
            raise PipelineError(
                f"{op_name} requires input '{key}' but it was not found. "
                f"Available keys: {available}"
            )


def _process_sample(
    sample: MutableMapping[str, mx.array],
    ops: List,
    state: Optional[MutableMapping[str, object]] = None,
    mode: Optional[str] = None,
) -> MutableMapping[str, mx.array]:
    """Apply Ops to a single sample.

    Args:
        sample: Dictionary of sample data.
        ops: List of ops to apply.
        state: Optional state dictionary.
        mode: Current execution mode ("train", "eval", etc.).

    Returns:
        Processed sample dictionary.

    Raises:
        PipelineError: If processing fails.
    """
    if state is None:
        state = {"mode": mode}
    else:
        state["mode"] = mode

    for i, op in enumerate(ops):
        # Check if op should run in current mode
        if hasattr(op, "should_run") and not op.should_run(mode):
            continue

        op_name = f"Op {i} ({op.__class__.__name__})"

        # Validate inputs
        _validate_op_inputs(op, sample, op_name)

        try:
            inp = sample[op.inputs[0]] if len(op.inputs) == 1 else [sample[k] for k in op.inputs]
            out = op.forward(inp, state)
        except Exception as e:
            raise PipelineError(f"{op_name}: Processing failed with error: {e}") from e

        if op.outputs:
            if len(op.outputs) == 1:
                sample[op.outputs[0]] = out
            else:
                for k, v in zip(op.outputs, out):
                    sample[k] = v

    return sample


class Pipeline:
    """Hold datasets and preprocessing operations.

    Args:
        train_data: Training dataset (iterable of samples).
        eval_data: Optional evaluation dataset.
        batch_size: Number of samples per batch.
        ops: List of preprocessing operations to apply.
        num_process: Number of worker processes for prefetching.
            If 0, no prefetching. If None, uses CPU count.

    Example:
        >>> pipeline = Pipeline(
        ...     train_data=train_ds,
        ...     eval_data=eval_ds,
        ...     batch_size=32,
        ...     ops=[Minmax("x", "x")]
        ... )
        >>> for batch in pipeline.get_loader("train"):
        ...     print(batch["x"].shape)
    """

    def __init__(
        self,
        train_data: Iterable,
        eval_data: Optional[Iterable] = None,
        batch_size: int = 32,
        ops: Optional[Iterable] = None,
        num_process: Optional[int] = 0
    ) -> None:
        self.train_data: Iterable = train_data
        self.eval_data: Optional[Iterable] = eval_data
        self.batch_size: int = batch_size
        self.ops: List = list(ops or [])
        self.num_process: int = mp.cpu_count() if num_process is None else num_process

    def _apply_ops(
        self,
        batch: MutableMapping[str, mx.array],
        mode: Optional[str] = None
    ) -> MutableMapping[str, mx.array]:
        """Apply ops to a batch of data.

        Args:
            batch: Dictionary of batched arrays.
            mode: Current execution mode ("train", "eval", etc.).

        Returns:
            Processed batch dictionary.
        """
        state: MutableMapping[str, object] = {"mode": mode}

        for i, op in enumerate(self.ops):
            # Check if op should run in current mode
            if hasattr(op, "should_run") and not op.should_run(mode):
                continue

            op_name = f"Op {i} ({op.__class__.__name__})"

            # Validate inputs
            _validate_op_inputs(op, batch, op_name)

            try:
                inp = batch[op.inputs[0]] if len(op.inputs) == 1 else [batch[k] for k in op.inputs]
                out = op.forward(inp, state)
            except Exception as e:
                raise PipelineError(f"{op_name}: Batch processing failed: {e}") from e

            if op.outputs:
                if len(op.outputs) == 1:
                    batch[op.outputs[0]] = out
                else:
                    for k_out, v_out in zip(op.outputs, out):
                        batch[k_out] = v_out

        return batch

    def _loader(
        self,
        dataset: Iterable,
        mode: str = "train",
        shuffle: bool = False
    ) -> Iterator[MutableMapping[str, object]]:
        """Create an iterator over processed batches.

        Args:
            dataset: The dataset to iterate over.
            mode: Current execution mode ("train", "eval", etc.).
            shuffle: Whether to shuffle the data.

        Yields:
            Processed batch dictionaries.
        """
        import numpy as np

        if isinstance(dataset, dx.Buffer):
            # MLX Data buffer - use streaming
            buffer = dataset
            if shuffle:
                buffer = buffer.shuffle()

            def transform(sample: MutableMapping[str, object]) -> MutableMapping[str, object]:
                mx_sample = {k: mx.array(v) for k, v in sample.items()}
                return _process_sample(mx_sample, self.ops, {}, mode=mode)

            stream = buffer.sample_transform(transform)
            stream = stream.batch(self.batch_size)
            if self.num_process and self.num_process > 0:
                stream = stream.ordered_prefetch(self.batch_size * 2, self.num_process)

            for batch in stream:
                yield {k: mx.array(v) for k, v in batch.items()}

        elif hasattr(dataset, "data") and hasattr(dataset, "__len__"):
            # In-memory dataset with .data attribute (like MLXDataset)
            try:
                arrays = {k: mx.array(v) for k, v in dataset.data.items()}
            except Exception as e:
                raise PipelineError(f"Failed to convert dataset.data to arrays: {e}") from e

            size = len(dataset)
            if size == 0:
                return  # Empty dataset

            # Create index array for shuffling
            indices = np.arange(size)
            if shuffle:
                np.random.shuffle(indices)

            for start in range(0, size, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                batch = {k: v[batch_indices] for k, v in arrays.items()}
                batch = self._apply_ops(batch, mode=mode)
                yield batch

        else:
            # Generic iterable - convert to buffer
            try:
                data_list = list(dataset)
            except Exception as e:
                raise PipelineError(f"Failed to convert dataset to list: {e}") from e

            if len(data_list) == 0:
                return  # Empty dataset

            if shuffle:
                import random
                data_list = data_list.copy()
                random.shuffle(data_list)

            buffer = dx.buffer_from_vector(data_list)

            def transform(sample: MutableMapping[str, object]) -> MutableMapping[str, object]:
                mx_sample = {k: mx.array(v) for k, v in sample.items()}
                return _process_sample(mx_sample, self.ops, {}, mode=mode)

            stream = buffer.sample_transform(transform)
            stream = stream.batch(self.batch_size)
            if self.num_process and self.num_process > 0:
                stream = stream.ordered_prefetch(self.batch_size * 2, self.num_process)

            for batch in stream:
                yield {k: mx.array(v) for k, v in batch.items()}

    def get_loader(
        self,
        mode: str = "train",
        shuffle: Optional[bool] = None
    ) -> Iterator[MutableMapping[str, object]]:
        """Get a data loader for the specified mode.

        Args:
            mode: Either 'train' or 'eval'.
            shuffle: Whether to shuffle the data. If None, shuffles for train mode only.

        Returns:
            Iterator yielding batch dictionaries.

        Raises:
            ValueError: If mode is invalid or dataset not available.
        """
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got '{mode}'")

        dataset = self.train_data if mode == "train" else self.eval_data

        if dataset is None:
            raise ValueError(f"No dataset available for mode '{mode}'")

        # Default: shuffle for training, no shuffle for eval
        if shuffle is None:
            shuffle = (mode == "train")

        return self._loader(dataset, mode=mode, shuffle=shuffle)
