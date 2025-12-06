"""A simple in-memory dataset backed by MLX arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping

if TYPE_CHECKING:
    import mlx.core as mx

import mlx.core as mx


class MLXDataset:
    """Dataset storing MLX arrays in memory.

    Args:
        data: Dictionary mapping keys to arrays. All arrays must have the same
            length (first dimension).

    Raises:
        ValueError: If data is empty or arrays have different lengths.
    """

    def __init__(self, data: Mapping[str, mx.array]) -> None:
        if not data:
            raise ValueError("Cannot create MLXDataset with empty data")

        self.data: Mapping[str, mx.array] = {k: mx.array(v) for k, v in data.items()}

        # Validate all arrays have the same size
        sizes = [len(v) for v in self.data.values()]
        if len(set(sizes)) > 1:
            size_info = {k: len(v) for k, v in self.data.items()}
            raise ValueError(f"All arrays must have the same length. Got: {size_info}")

        self.size: int = sizes[0]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        return {k: v[idx] for k, v in self.data.items()}

