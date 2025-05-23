"""A simple in-memory dataset backed by MLX arrays."""

from __future__ import annotations

from typing import Dict, Mapping

import mlx.core as mx

Array = mx.array


class MLXDataset:
    """Dataset storing MLX arrays in memory."""

    def __init__(self, data: Mapping[str, Array]):
        self.data: Mapping[str, Array] = {k: mx.array(v) for k, v in data.items()}
        self.size: int = len(next(iter(self.data.values())))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Array]:
        return {k: v[idx] for k, v in self.data.items()}

