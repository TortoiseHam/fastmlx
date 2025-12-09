"""NumPy-based in-memory datasets."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import mlx.core as mx
import numpy as np


class NumpyDataset:
    """Dataset created from a dictionary of NumPy arrays or lists.

    All arrays/lists must have the same length (first dimension).

    Args:
        data: Dictionary mapping keys to NumPy arrays or lists.

    Example:
        >>> data = {"x": np.random.randn(100, 28, 28), "y": np.arange(100)}
        >>> dataset = NumpyDataset(data)
        >>> len(dataset)
        100
        >>> sample = dataset[0]
        >>> sample["x"].shape
        (28, 28)
    """

    def __init__(self, data: Dict[str, Union[np.ndarray, List]]) -> None:
        if not data:
            self._data: Dict[str, np.ndarray] = {}
            self._length = 0
            return

        # Validate and convert to numpy arrays
        self._data = {}
        self._length: Optional[int] = None

        for key, value in data.items():
            if isinstance(value, list):
                arr = np.array(value)
            elif isinstance(value, np.ndarray):
                arr = value
            else:
                raise ValueError(f"Value for key '{key}' must be a numpy array or list, got {type(value)}")

            if self._length is None:
                self._length = len(arr)
            elif len(arr) != self._length:
                raise ValueError(
                    f"All arrays must have the same length. "
                    f"Expected {self._length}, got {len(arr)} for key '{key}'"
                )

            self._data[key] = arr

        if self._length is None:
            self._length = 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self._length}")

        return {key: mx.array(arr[idx]) for key, arr in self._data.items()}

    @property
    def keys(self) -> List[str]:
        """Return list of data keys."""
        return list(self._data.keys())

    @classmethod
    def from_arrays(cls, **arrays: np.ndarray) -> "NumpyDataset":
        """Create dataset from keyword arguments.

        Example:
            >>> dataset = NumpyDataset.from_arrays(
            ...     x=np.random.randn(100, 28, 28),
            ...     y=np.arange(100)
            ... )
        """
        return cls(arrays)


class PickleDataset:
    """Dataset loaded from a pickle file.

    The pickle file should contain a dictionary with keys mapping to
    arrays or lists, or a list of dictionaries (one per sample).

    Args:
        path: Path to the pickle file.
        keys: Optional list of keys to load. If None, loads all keys.

    Example:
        >>> dataset = PickleDataset("data.pkl")
        >>> sample = dataset[0]
    """

    def __init__(self, path: str, keys: Optional[List[str]] = None) -> None:
        self.path = Path(path)
        self._keys = keys

        with open(self.path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            # Dictionary format: {"x": array, "y": array}
            if keys:
                data = {k: data[k] for k in keys if k in data}
            self._data = data
            self._length = len(next(iter(data.values())))
            self._format = "dict"
        elif isinstance(data, list):
            # List format: [{"x": ..., "y": ...}, ...]
            self._samples = data
            self._length = len(data)
            self._format = "list"
        else:
            raise ValueError(f"Pickle must contain dict or list, got {type(data)}")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range")

        if self._format == "dict":
            result = {}
            for key, arr in self._data.items():
                val = arr[idx]
                if isinstance(val, np.ndarray):
                    result[key] = mx.array(val)
                elif isinstance(val, (int, float)):
                    result[key] = mx.array([val])
                else:
                    result[key] = val
            return result
        else:
            sample = self._samples[idx]
            result = {}
            for key, val in sample.items():
                if isinstance(val, np.ndarray):
                    result[key] = mx.array(val)
                elif isinstance(val, (int, float)):
                    result[key] = mx.array([val])
                else:
                    result[key] = val
            return result


class InMemoryDataset:
    """Base class for in-memory datasets.

    Stores all data in memory for fast access.

    Args:
        samples: List of sample dictionaries.

    Example:
        >>> samples = [{"x": np.array([1, 2]), "y": 0}, {"x": np.array([3, 4]), "y": 1}]
        >>> dataset = InMemoryDataset(samples)
    """

    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        sample = self._samples[idx]
        result = {}
        for key, val in sample.items():
            if isinstance(val, np.ndarray):
                result[key] = mx.array(val)
            elif isinstance(val, mx.array):
                result[key] = val
            elif isinstance(val, (int, float)):
                result[key] = mx.array([val])
            else:
                result[key] = val
        return result

    def shuffle(self) -> None:
        """Shuffle the dataset in place."""
        np.random.shuffle(self._samples)

    def split(self, ratio: float = 0.8) -> tuple["InMemoryDataset", "InMemoryDataset"]:
        """Split dataset into train and validation sets.

        Args:
            ratio: Fraction of data for training set.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        n_train = int(len(self._samples) * ratio)
        train_samples = self._samples[:n_train]
        val_samples = self._samples[n_train:]
        return InMemoryDataset(train_samples), InMemoryDataset(val_samples)
