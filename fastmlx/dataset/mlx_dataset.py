"""A simple in-memory dataset backed by numpy arrays.

Data is stored as numpy arrays to avoid Metal buffer allocation limits when
iterating through large datasets. Conversion to MLX arrays happens at batch
time in the Pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Union

import numpy as np

# Accept both numpy and MLX arrays as input
ArrayLike = Union[np.ndarray, Any]  # Any covers mx.array without import


class MLXDataset:
    """Dataset storing arrays in memory as numpy.

    Data is stored internally as numpy arrays to avoid Metal buffer allocation
    limits. Each ``__getitem__`` call returns numpy arrays, and conversion to
    MLX arrays happens at batch time in the Pipeline's collate step.

    Args:
        data: Dictionary mapping keys to arrays (numpy or MLX). All arrays
            must have the same length (first dimension).

    Raises:
        ValueError: If data is empty or arrays have different lengths.
    """

    def __init__(self, data: Mapping[str, ArrayLike]) -> None:
        if not data:
            raise ValueError("Cannot create MLXDataset with empty data")

        # Convert all arrays to numpy to avoid Metal allocation explosion
        # when iterating through the dataset
        self.data: Dict[str, np.ndarray] = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                self.data[k] = v
            else:
                # Handles mx.array and other array-like objects
                self.data[k] = np.array(v)

        # Validate all arrays have the same size
        sizes = [len(v) for v in self.data.values()]
        if len(set(sizes)) > 1:
            size_info = {k: len(v) for k, v in self.data.items()}
            raise ValueError(f"All arrays must have the same length. Got: {size_info}")

        self.size: int = sizes[0]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample by index.

        Returns numpy arrays to avoid Metal buffer allocation limits.
        Conversion to MLX arrays happens at batch time.
        """
        return {k: v[idx] for k, v in self.data.items()}

