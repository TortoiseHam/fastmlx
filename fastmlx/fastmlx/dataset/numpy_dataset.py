"""A simple in-memory dataset backed by NumPy arrays."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np


class NumpyDataset:
    def __init__(self, data: Mapping[str, np.ndarray]):
        self.data: Mapping[str, np.ndarray] = data
        self.size: int = len(next(iter(data.values())))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {k: v[idx] for k, v in self.data.items()}
