"""Generator-based Dataset implementation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Generator, Iterator, Optional

import mlx.core as mx


class GeneratorDataset:
    """Dataset that wraps a generator function for lazy data loading.

    Args:
        generator_fn: A function that returns a generator yielding samples.
                     Each sample should be a dict with string keys and array values.
        size: Total number of samples. If None, the dataset is unbounded.
        repeat: If True, restart the generator when exhausted.

    Example:
        >>> def my_generator():
        ...     for i in range(1000):
        ...         yield {"x": mx.array([i]), "y": mx.array([i % 10])}
        >>> dataset = GeneratorDataset(my_generator, size=1000)
        >>> print(len(dataset))
        1000
    """

    def __init__(
        self,
        generator_fn: Callable[[], Generator[Dict[str, mx.array], None, None]],
        size: Optional[int] = None,
        repeat: bool = True
    ) -> None:
        self.generator_fn = generator_fn
        self._size = size
        self.repeat = repeat
        self._generator: Optional[Generator] = None
        self._buffer: Dict[int, Dict[str, mx.array]] = {}
        self._next_idx: int = 0

    def __len__(self) -> int:
        if self._size is None:
            raise ValueError("Cannot get length of unbounded GeneratorDataset")
        return self._size

    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        self._generator = self.generator_fn()
        return self

    def __next__(self) -> Dict[str, mx.array]:
        if self._generator is None:
            self._generator = self.generator_fn()

        try:
            return next(self._generator)
        except StopIteration:
            if self.repeat:
                self._generator = self.generator_fn()
                return next(self._generator)
            raise

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        # Check buffer first
        if idx in self._buffer:
            return self._buffer[idx]

        # Need to iterate to get to this index
        if self._generator is None:
            self._generator = self.generator_fn()
            self._next_idx = 0

        # If requested index is before current position, restart
        if idx < self._next_idx:
            self._generator = self.generator_fn()
            self._next_idx = 0
            self._buffer.clear()

        # Iterate until we reach the requested index
        while self._next_idx <= idx:
            try:
                sample = next(self._generator)
                self._buffer[self._next_idx] = sample
                self._next_idx += 1
            except StopIteration:
                if self.repeat:
                    self._generator = self.generator_fn()
                    sample = next(self._generator)
                    self._buffer[self._next_idx] = sample
                    self._next_idx += 1
                else:
                    raise IndexError(f"Index {idx} out of range")

        return self._buffer[idx]


class BatchDataset:
    """Dataset that batches samples from another dataset.

    Args:
        dataset: The source dataset.
        batch_size: Number of samples per batch.
        drop_last: If True, drop the last incomplete batch.
        shuffle: If True, shuffle indices before batching.

    Example:
        >>> from fastmlx.dataset import MLXDataset
        >>> source = MLXDataset({"x": mx.arange(100), "y": mx.arange(100)})
        >>> batched = BatchDataset(source, batch_size=32)
        >>> print(len(batched))
        3
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self._indices: Optional[list] = None

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _get_indices(self) -> list:
        """Get shuffled or sequential indices."""
        import numpy as np
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        return indices

    def __getitem__(self, batch_idx: int) -> Dict[str, mx.array]:
        if self._indices is None:
            self._indices = self._get_indices()

        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.dataset))

        if start >= len(self.dataset):
            raise IndexError(f"Batch index {batch_idx} out of range")

        indices = self._indices[start:end]

        # Collect samples
        samples = [self.dataset[i] for i in indices]

        # Stack into batch
        batch: Dict[str, mx.array] = {}
        for key in samples[0].keys():
            batch[key] = mx.stack([s[key] for s in samples])

        return batch

    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        self._indices = self._get_indices()
        for i in range(len(self)):
            yield self[i]


class CombinedDataset:
    """Combine multiple datasets by concatenation.

    Args:
        datasets: List of datasets to combine.

    Example:
        >>> ds1 = MLXDataset({"x": mx.arange(100)})
        >>> ds2 = MLXDataset({"x": mx.arange(100, 200)})
        >>> combined = CombinedDataset([ds1, ds2])
        >>> print(len(combined))
        200
    """

    def __init__(self, datasets: list) -> None:
        self.datasets = datasets
        self._cumulative_sizes = []

        cumsum = 0
        for ds in datasets:
            cumsum += len(ds)
            self._cumulative_sizes.append(cumsum)

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        if idx < 0:
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        # Find which dataset this index belongs to
        for ds_idx, cumsum in enumerate(self._cumulative_sizes):
            if idx < cumsum:
                if ds_idx == 0:
                    local_idx = idx
                else:
                    local_idx = idx - self._cumulative_sizes[ds_idx - 1]
                return self.datasets[ds_idx][local_idx]

        raise IndexError(f"Index {idx} out of range")


class InterleaveDataset:
    """Interleave samples from multiple datasets.

    Unlike CombinedDataset which concatenates, this alternates between datasets.

    Args:
        datasets: List of datasets to interleave.
        cycle: If True, cycle through smaller datasets to match the largest.

    Example:
        >>> ds1 = MLXDataset({"x": mx.array([1, 2, 3])})
        >>> ds2 = MLXDataset({"x": mx.array([4, 5, 6])})
        >>> interleaved = InterleaveDataset([ds1, ds2])
        >>> [interleaved[i]["x"].item() for i in range(6)]
        [1, 4, 2, 5, 3, 6]
    """

    def __init__(self, datasets: list, cycle: bool = False) -> None:
        self.datasets = datasets
        self.cycle = cycle
        self.num_datasets = len(datasets)

        if cycle:
            self.max_len = max(len(ds) for ds in datasets)
            self._size = self.max_len * self.num_datasets
        else:
            self.min_len = min(len(ds) for ds in datasets)
            self._size = self.min_len * self.num_datasets

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        if idx < 0:
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        ds_idx = idx % self.num_datasets
        sample_idx = idx // self.num_datasets

        if self.cycle:
            # Cycle through shorter datasets
            sample_idx = sample_idx % len(self.datasets[ds_idx])

        return self.datasets[ds_idx][sample_idx]
