"""Numpy based operations."""

from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
from ..op import Op

class ExpandDims(Op):
    def __init__(self, inputs: str, outputs: str, axis: int = 0) -> None:
        super().__init__(inputs, outputs)
        self.axis = axis

    def forward(self, data: np.ndarray, state: MutableMapping[str, Any]) -> np.ndarray:
        return np.expand_dims(data, axis=self.axis)

class Minmax(Op):
    def __init__(self, inputs: str, outputs: str) -> None:
        super().__init__(inputs, outputs)

    def forward(self, data: np.ndarray, state: MutableMapping[str, Any]) -> np.ndarray:
        return data.astype(np.float32) / 255.0
