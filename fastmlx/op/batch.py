"""Batch operation for collating samples into batches."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence

import mlx.core as mx
import numpy as np

from .op import Op


class Batch(Op):
    """Marks the transition from sample-level to batch-level processing.

    The Batch op serves as a marker in the Pipeline's op list that separates:
    - Sample ops (before Batch): operate on individual samples, numpy arrays
    - Batch ops (after Batch): operate on batched data, MLX arrays

    It also configures how samples are collated into batches, including support
    for padding variable-length sequences.

    Args:
        batch_size: Number of samples per batch.
        drop_last: If True, drop the last incomplete batch. Default False.
        pad_value: Value to use for padding variable-length sequences. If None,
            all samples must have the same shape (will raise error otherwise).
        collate_fn: Custom function to collate samples. Signature:
            (samples: List[Dict[str, np.ndarray]]) -> Dict[str, mx.array]
            If provided, pad_value is ignored.
        pad_keys: Keys to apply padding to. If None, pads all keys with
            variable lengths. Only used when pad_value is set.

    Example:
        Basic usage with fixed-size data:
        >>> pipeline = Pipeline(
        ...     train_data=dataset,
        ...     ops=[
        ...         RandomCrop(inputs="x", outputs="x"),  # Sample op
        ...         Batch(batch_size=32),
        ...         Normalize(inputs="x", outputs="x"),   # Batch op
        ...     ]
        ... )

        With padding for variable-length sequences:
        >>> pipeline = Pipeline(
        ...     train_data=text_dataset,
        ...     ops=[
        ...         Tokenize(inputs="text", outputs="tokens"),
        ...         Batch(batch_size=32, pad_value=0),  # Pads to max length
        ...         Embedding(inputs="tokens", outputs="embedded"),
        ...     ]
        ... )

        With custom collation:
        >>> def custom_collate(samples):
        ...     # Custom logic for complex data structures
        ...     return {
        ...         "x": mx.array(np.stack([s["x"] for s in samples])),
        ...         "mask": mx.array(np.stack([s["mask"] for s in samples])),
        ...     }
        >>> pipeline = Pipeline(
        ...     train_data=dataset,
        ...     ops=[Batch(batch_size=32, collate_fn=custom_collate)]
        ... )
    """

    def __init__(
        self,
        batch_size: int = 32,
        drop_last: bool = False,
        pad_value: Optional[float] = None,
        collate_fn: Optional[Callable[[List[Dict[str, Any]]], Dict[str, mx.array]]] = None,
        pad_keys: Optional[Sequence[str]] = None,
    ) -> None:
        # Batch op doesn't process inputs/outputs like regular ops
        super().__init__(inputs=[], outputs=[])
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.pad_value = pad_value
        self.pad_keys = set(pad_keys) if pad_keys else None
        self._collate_fn = collate_fn

    def collate(self, samples: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        """Collate a list of samples into a batch.

        Args:
            samples: List of sample dictionaries.

        Returns:
            Batch dictionary with MLX arrays.
        """
        if self._collate_fn is not None:
            return self._collate_fn(samples)

        if not samples:
            return {}

        # Get all keys from first sample
        keys = samples[0].keys()
        batch = {}

        for key in keys:
            values = [s[key] for s in samples]
            batch[key] = self._collate_key(key, values)

        return batch

    def _collate_key(self, key: str, values: List[Any]) -> mx.array:
        """Collate values for a single key.

        Args:
            key: The key name (for error messages and pad_keys check).
            values: List of values to collate.

        Returns:
            Batched MLX array.
        """
        # Convert to numpy arrays if needed
        arrays = []
        for v in values:
            if isinstance(v, mx.array):
                arrays.append(np.array(v))
            elif isinstance(v, np.ndarray):
                arrays.append(v)
            else:
                arrays.append(np.array(v))

        # Check if shapes match
        shapes = [a.shape for a in arrays]
        shapes_match = all(s == shapes[0] for s in shapes)

        if shapes_match:
            # Simple stack
            stacked = np.stack(arrays, axis=0)
            return mx.array(stacked)

        # Shapes don't match - need padding or error
        should_pad = self.pad_keys is None or key in self.pad_keys

        if self.pad_value is None or not should_pad:
            raise ValueError(
                f"Variable-length data for key '{key}' but no padding configured. "
                f"Shapes: {shapes[:5]}{'...' if len(shapes) > 5 else ''}. "
                f"Set pad_value in Batch() to enable padding."
            )

        return self._pad_and_stack(arrays)

    def _pad_and_stack(self, arrays: List[np.ndarray]) -> mx.array:
        """Pad arrays to max shape and stack.

        Args:
            arrays: List of arrays with potentially different shapes.

        Returns:
            Padded and stacked MLX array.
        """
        # Find max shape for each dimension
        ndim = arrays[0].ndim
        max_shape = [max(a.shape[i] for a in arrays) for i in range(ndim)]

        # Pad each array
        padded = []
        for arr in arrays:
            if arr.shape == tuple(max_shape):
                padded.append(arr)
            else:
                # Create padded array
                pad_arr = np.full(max_shape, self.pad_value, dtype=arr.dtype)
                # Copy original data
                slices = tuple(slice(0, s) for s in arr.shape)
                pad_arr[slices] = arr
                padded.append(pad_arr)

        stacked = np.stack(padded, axis=0)
        return mx.array(stacked)

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Any:
        """Batch op doesn't use forward() - collation happens in Pipeline."""
        raise RuntimeError(
            "Batch.forward() should not be called directly. "
            "The Pipeline handles batching via Batch.collate()."
        )


class DynamicBatch(Batch):
    """Batch op that dynamically adjusts batch size based on sample sizes.

    Useful for variable-length sequences where you want to maximize GPU
    utilization by fitting as many tokens/elements as possible per batch.

    Args:
        max_tokens: Maximum total tokens/elements per batch.
        max_batch_size: Maximum number of samples per batch (upper bound).
        size_fn: Function to compute size of a sample. Default counts elements
            in the first array.
        drop_last: If True, drop the last incomplete batch.
        pad_value: Value for padding. Required for variable-length data.
        collate_fn: Custom collation function.

    Example:
        >>> pipeline = Pipeline(
        ...     train_data=text_dataset,
        ...     ops=[
        ...         Tokenize(inputs="text", outputs="tokens"),
        ...         DynamicBatch(
        ...             max_tokens=4096,
        ...             max_batch_size=64,
        ...             size_fn=lambda s: len(s["tokens"]),
        ...             pad_value=0,
        ...         ),
        ...     ]
        ... )
    """

    def __init__(
        self,
        max_tokens: int,
        max_batch_size: int = 128,
        size_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
        drop_last: bool = False,
        pad_value: Optional[float] = 0,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            batch_size=max_batch_size,  # Upper bound
            drop_last=drop_last,
            pad_value=pad_value,
            collate_fn=collate_fn,
        )
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self._size_fn = size_fn or self._default_size_fn

    @staticmethod
    def _default_size_fn(sample: Dict[str, Any]) -> int:
        """Default: count elements in first array."""
        for v in sample.values():
            if hasattr(v, "size"):
                return v.size
            if hasattr(v, "__len__"):
                return len(v)
        return 1

    def compute_batch_indices(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[List[int]]:
        """Compute batch groupings based on sample sizes.

        Args:
            samples: List of all samples.

        Returns:
            List of lists, where each inner list contains indices for one batch.
        """
        # Sort by size for better packing
        sizes = [(i, self._size_fn(s)) for i, s in enumerate(samples)]
        sizes.sort(key=lambda x: x[1], reverse=True)

        batches = []
        current_batch: List[int] = []
        max_size_in_batch = 0

        for idx, size in sizes:
            # Would adding this sample exceed limits?
            new_max_size = max(max_size_in_batch, size)
            new_batch_size = len(current_batch) + 1
            # With padding, total tokens = batch_size * max_seq_len
            new_tokens = new_batch_size * new_max_size

            if current_batch and (
                new_tokens > self.max_tokens or
                new_batch_size > self.max_batch_size
            ):
                # Start new batch
                batches.append(current_batch)
                current_batch = [idx]
                max_size_in_batch = size
            else:
                current_batch.append(idx)
                max_size_in_batch = new_max_size

        if current_batch and (not self.drop_last or len(current_batch) == self.max_batch_size):
            batches.append(current_batch)

        return batches
