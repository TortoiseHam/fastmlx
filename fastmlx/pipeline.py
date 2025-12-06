"""Data loading and preprocessing pipeline."""

from __future__ import annotations

import multiprocessing as mp
import sys
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Union,
)

import mlx.core as mx
import numpy as np

# mlx-data only available on macOS (Apple Silicon)
if sys.platform == "darwin":
    import mlx.data as dx
else:
    dx = None  # type: ignore

from .op.batch import Batch, DynamicBatch
from .op.filtered_data import FilteredData


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


def _apply_sample_op(
    op: Any,
    sample: MutableMapping[str, Any],
    state: MutableMapping[str, Any],
    op_index: int,
) -> Union[MutableMapping[str, Any], FilteredData]:
    """Apply a single op to a sample.

    Args:
        op: The operation to apply.
        sample: The sample dictionary.
        state: State dictionary.
        op_index: Index of op (for error messages).

    Returns:
        Processed sample or FilteredData if sample should be dropped.
    """
    op_name = f"Op {op_index} ({op.__class__.__name__})"

    # Check if op should run in current mode
    mode = state.get("mode")
    if hasattr(op, "should_run") and not op.should_run(mode):
        return sample

    # Validate inputs
    _validate_op_inputs(op, sample, op_name)

    try:
        if len(op.inputs) == 1:
            inp = sample[op.inputs[0]]
        else:
            inp = [sample[k] for k in op.inputs]

        out = op.forward(inp, state)
    except Exception as e:
        raise PipelineError(f"{op_name}: Processing failed: {e}") from e

    # Check if sample should be filtered
    if isinstance(out, FilteredData):
        return out

    # Store outputs
    if op.outputs:
        if len(op.outputs) == 1:
            sample[op.outputs[0]] = out
        else:
            if not isinstance(out, (list, tuple)):
                raise PipelineError(
                    f"{op_name}: Expected {len(op.outputs)} outputs but got single value"
                )
            for k, v in zip(op.outputs, out):
                sample[k] = v

    return sample


def _process_sample(
    sample: MutableMapping[str, Any],
    ops: Sequence[Any],
    state: MutableMapping[str, Any],
) -> Union[MutableMapping[str, Any], FilteredData]:
    """Apply sample ops to a single sample.

    Args:
        sample: Dictionary of sample data.
        ops: List of sample-level ops to apply.
        state: State dictionary with 'mode' key.

    Returns:
        Processed sample or FilteredData if filtered.
    """
    for i, op in enumerate(ops):
        result = _apply_sample_op(op, sample, state, i)
        if isinstance(result, FilteredData):
            return result
        sample = result

    return sample


def _apply_batch_ops(
    batch: MutableMapping[str, mx.array],
    ops: Sequence[Any],
    state: MutableMapping[str, Any],
) -> MutableMapping[str, mx.array]:
    """Apply batch-level ops to a batch.

    Args:
        batch: Dictionary of batched MLX arrays.
        ops: List of batch-level ops to apply.
        state: State dictionary with 'mode' key.

    Returns:
        Processed batch.
    """
    for i, op in enumerate(ops):
        op_name = f"BatchOp {i} ({op.__class__.__name__})"
        mode = state.get("mode")

        # Check if op should run in current mode
        if hasattr(op, "should_run") and not op.should_run(mode):
            continue

        _validate_op_inputs(op, batch, op_name)

        try:
            if len(op.inputs) == 1:
                inp = batch[op.inputs[0]]
            else:
                inp = [batch[k] for k in op.inputs]

            # Use forward_batch if available, otherwise forward
            if hasattr(op, "forward_batch"):
                out = op.forward_batch(inp, state)
            else:
                out = op.forward(inp, state)
        except Exception as e:
            raise PipelineError(f"{op_name}: Batch processing failed: {e}") from e

        if op.outputs:
            if len(op.outputs) == 1:
                batch[op.outputs[0]] = out
            else:
                for k, v in zip(op.outputs, out):
                    batch[k] = v

    return batch


class Pipeline:
    """Data loading and preprocessing pipeline with sample and batch operations.

    The Pipeline orchestrates data flow through three stages:
    1. Sample ops: Applied to individual samples (numpy arrays)
    2. Batching: Collates samples into batches (converts to MLX arrays)
    3. Batch ops: Applied to batched data (MLX arrays)

    Ops are split based on the Batch op position in the ops list:
    - Ops before Batch: sample-level (can filter, work on numpy)
    - Ops after Batch: batch-level (work on MLX arrays, e.g., MixUp)

    Args:
        train_data: Training dataset (iterable of sample dicts).
        eval_data: Optional evaluation dataset.
        batch_size: Default batch size (used if no Batch op specified).
        ops: List of preprocessing operations.
        num_process: Number of workers for prefetching (0 = no prefetching).

    Example:
        Basic pipeline:
        >>> pipeline = Pipeline(
        ...     train_data=train_ds,
        ...     eval_data=eval_ds,
        ...     batch_size=32,
        ...     ops=[Normalize(inputs="x", outputs="x")]
        ... )

        With explicit batching and filtering:
        >>> pipeline = Pipeline(
        ...     train_data=train_ds,
        ...     ops=[
        ...         # Sample ops (before Batch)
        ...         DropSmallImages(min_size=64),  # Returns FilteredData
        ...         RandomCrop(inputs="x", outputs="x"),
        ...         HorizontalFlip(inputs="x", outputs="x"),
        ...
        ...         # Batching configuration
        ...         Batch(batch_size=32, drop_last=True),
        ...
        ...         # Batch ops (after Batch)
        ...         Normalize(inputs="x", outputs="x"),
        ...         MixUp(inputs=["x", "y"], outputs=["x", "y"]),
        ...     ]
        ... )

        With padding for variable-length data:
        >>> pipeline = Pipeline(
        ...     train_data=text_ds,
        ...     ops=[
        ...         Tokenize(inputs="text", outputs="tokens"),
        ...         Batch(batch_size=32, pad_value=0),  # Pads sequences
        ...         Embedding(inputs="tokens", outputs="x"),
        ...     ]
        ... )
    """

    def __init__(
        self,
        train_data: Iterable[Dict[str, Any]],
        eval_data: Optional[Iterable[Dict[str, Any]]] = None,
        batch_size: int = 32,
        ops: Optional[Iterable[Any]] = None,
        num_process: Optional[int] = 0,
    ) -> None:
        self.train_data = train_data
        self.eval_data = eval_data
        self.num_process = mp.cpu_count() if num_process is None else num_process

        # Parse ops into sample ops, batch op, batch ops
        ops_list = list(ops or [])
        self.sample_ops, self.batch_op, self.batch_ops = self._split_ops(ops_list)

        # Create default Batch op if none specified
        if self.batch_op is None:
            self.batch_op = Batch(batch_size=batch_size)

    def _split_ops(
        self,
        ops: List[Any],
    ) -> tuple[List[Any], Optional[Batch], List[Any]]:
        """Split ops into sample ops, batch op, and batch ops.

        Args:
            ops: Full list of operations.

        Returns:
            Tuple of (sample_ops, batch_op, batch_ops).
        """
        sample_ops: List[Any] = []
        batch_op: Optional[Batch] = None
        batch_ops: List[Any] = []

        for op in ops:
            if isinstance(op, (Batch, DynamicBatch)):
                if batch_op is not None:
                    raise ValueError(
                        "Only one Batch op allowed per Pipeline. "
                        f"Found multiple: {batch_op}, {op}"
                    )
                batch_op = op
            elif batch_op is None:
                sample_ops.append(op)
            else:
                batch_ops.append(op)

        return sample_ops, batch_op, batch_ops

    def _sample_iterator(
        self,
        dataset: Iterable[Dict[str, Any]],
        mode: str,
        shuffle: bool,
    ) -> Iterator[Dict[str, Any]]:
        """Create an iterator that applies sample ops with filtering.

        Args:
            dataset: The dataset to iterate over.
            mode: Current mode ("train", "eval").
            shuffle: Whether to shuffle.

        Yields:
            Processed samples (FilteredData samples are skipped).
        """
        state: Dict[str, Any] = {"mode": mode}

        # Convert dataset to list for shuffling if needed
        if shuffle:
            data_list = list(dataset)
            np.random.shuffle(data_list)
            data_iter: Iterable[Dict[str, Any]] = data_list
        else:
            data_iter = dataset

        # Use MLX Data prefetching if requested and available
        if self.num_process > 0 and not shuffle:
            if dx is None:
                raise PipelineError(
                    "Parallel prefetching (num_process > 0) requires mlx-data, "
                    "which is only available on macOS. Set num_process=0 on Linux."
                )
            # MLX Data can parallelize sample transforms
            yield from self._mlx_data_sample_iterator(data_iter, state)
        else:
            # Simple sequential processing
            for sample in data_iter:
                # Ensure numpy arrays
                sample = self._to_numpy(sample)
                result = _process_sample(sample, self.sample_ops, state)

                if not isinstance(result, FilteredData):
                    yield result

    def _mlx_data_sample_iterator(
        self,
        dataset: Iterable[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Iterator[Dict[str, Any]]:
        """Use MLX Data for parallel sample processing.

        Args:
            dataset: The dataset.
            state: State dictionary.

        Yields:
            Processed samples.
        """
        # Convert to list for MLX Data buffer
        data_list = list(dataset)
        if not data_list:
            return

        buffer = dx.buffer_from_vector(data_list)

        def transform(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            sample = self._to_numpy(sample)
            result = _process_sample(sample, self.sample_ops, state)
            if isinstance(result, FilteredData):
                return None  # Will be filtered by stream filter
            return result

        stream = buffer.sample_transform(transform)
        stream = stream.ordered_prefetch(
            self.batch_op.batch_size * 2,
            self.num_process
        )

        for sample in stream:
            if sample is not None:
                yield sample

    def _to_numpy(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert sample values to numpy arrays.

        Args:
            sample: Sample dictionary.

        Returns:
            Sample with numpy arrays.
        """
        result = {}
        for k, v in sample.items():
            if isinstance(v, mx.array):
                result[k] = np.array(v)
            elif isinstance(v, np.ndarray):
                result[k] = v
            else:
                result[k] = np.array(v)
        return result

    def _batch_iterator(
        self,
        sample_iter: Iterator[Dict[str, Any]],
        mode: str,
    ) -> Iterator[Dict[str, mx.array]]:
        """Collect samples into batches and apply batch ops.

        Args:
            sample_iter: Iterator of processed samples.
            mode: Current mode.

        Yields:
            Batched and processed data.
        """
        state: Dict[str, Any] = {"mode": mode}
        batch_size = self.batch_op.batch_size
        drop_last = self.batch_op.drop_last

        # Handle DynamicBatch differently
        if isinstance(self.batch_op, DynamicBatch):
            yield from self._dynamic_batch_iterator(sample_iter, state)
            return

        # Regular batching
        samples: List[Dict[str, Any]] = []

        for sample in sample_iter:
            samples.append(sample)

            if len(samples) >= batch_size:
                batch = self.batch_op.collate(samples[:batch_size])
                batch = _apply_batch_ops(batch, self.batch_ops, state)
                yield batch
                samples = samples[batch_size:]

        # Handle remaining samples
        if samples and not drop_last:
            batch = self.batch_op.collate(samples)
            batch = _apply_batch_ops(batch, self.batch_ops, state)
            yield batch

    def _dynamic_batch_iterator(
        self,
        sample_iter: Iterator[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Iterator[Dict[str, mx.array]]:
        """Handle DynamicBatch which groups by token count.

        Args:
            sample_iter: Iterator of processed samples.
            state: State dictionary.

        Yields:
            Dynamically batched data.
        """
        # Collect all samples first (needed for size-based grouping)
        samples = list(sample_iter)
        if not samples:
            return

        # Get batch groupings
        batch_indices = self.batch_op.compute_batch_indices(samples)

        for indices in batch_indices:
            batch_samples = [samples[i] for i in indices]
            batch = self.batch_op.collate(batch_samples)
            batch = _apply_batch_ops(batch, self.batch_ops, state)
            yield batch

    def get_loader(
        self,
        mode: str = "train",
        shuffle: Optional[bool] = None,
    ) -> Iterator[Dict[str, mx.array]]:
        """Get a data loader for the specified mode.

        Args:
            mode: Either 'train' or 'eval'.
            shuffle: Whether to shuffle. Defaults to True for train, False for eval.

        Returns:
            Iterator yielding batch dictionaries with MLX arrays.

        Raises:
            ValueError: If mode is invalid or dataset unavailable.
        """
        if mode not in ("train", "eval", "test", "infer"):
            raise ValueError(f"mode must be 'train', 'eval', 'test', or 'infer', got '{mode}'")

        if mode == "train":
            dataset = self.train_data
        else:
            dataset = self.eval_data

        if dataset is None:
            raise ValueError(f"No dataset available for mode '{mode}'")

        # Default shuffle behavior
        if shuffle is None:
            shuffle = (mode == "train")

        # Create sample iterator (with filtering)
        sample_iter = self._sample_iterator(dataset, mode, shuffle)

        # Create batch iterator (with batch ops)
        return self._batch_iterator(sample_iter, mode)

    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over training data."""
        return self.get_loader("train")

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self.batch_op.batch_size

    def benchmark(
        self,
        mode: str = "train",
        num_batches: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Benchmark the pipeline throughput.

        Args:
            mode: Mode to benchmark.
            num_batches: Number of batches to time.
            warmup: Number of warmup batches.

        Returns:
            Dictionary with timing statistics.
        """
        import time

        loader = self.get_loader(mode)

        # Warmup
        for i, _ in enumerate(loader):
            if i >= warmup:
                break

        # Benchmark
        loader = self.get_loader(mode)
        times = []
        batch_sizes = []

        start = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            mx.eval(batch)  # Force evaluation
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            # Get actual batch size from first array
            for v in batch.values():
                batch_sizes.append(v.shape[0])
                break
            start = time.perf_counter()

        if not times:
            return {"error": "No batches produced"}

        total_samples = sum(batch_sizes)
        total_time = sum(times)

        return {
            "batches": len(times),
            "total_samples": total_samples,
            "total_time_sec": total_time,
            "samples_per_sec": total_samples / total_time if total_time > 0 else 0,
            "avg_batch_time_ms": (total_time / len(times)) * 1000,
            "avg_batch_size": total_samples / len(times),
        }

    def transform(
        self,
        sample: Dict[str, Any],
        mode: str = "infer",
    ) -> Dict[str, mx.array]:
        """Apply pipeline operations to a single sample.

        This is useful for inference on individual samples outside of training.
        Applies sample ops, batches (with batch size 1), and batch ops.

        Args:
            sample: A single sample dictionary.
            mode: Execution mode. Default "infer" for inference.

        Returns:
            Transformed sample as a batch of size 1 with MLX arrays.

        Example:
            >>> # During inference
            >>> sample = {"x": np.array([1, 2, 3])}
            >>> batch = pipeline.transform(sample, mode="infer")
            >>> prediction = model(batch["x"])
        """
        state: Dict[str, Any] = {"mode": mode}

        # Ensure numpy arrays
        sample = self._to_numpy(sample)

        # Apply sample ops
        result = _process_sample(sample, self.sample_ops, state)

        if isinstance(result, FilteredData):
            raise PipelineError(
                "Sample was filtered out by pipeline ops. "
                "Check your filtering conditions."
            )

        # Batch the single sample (creates batch of size 1)
        batch = self.batch_op.collate([result])

        # Apply batch ops
        batch = _apply_batch_ops(batch, self.batch_ops, state)

        return batch
