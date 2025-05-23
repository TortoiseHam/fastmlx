"""Array operations implemented with MLX."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

import numpy as np

import mlx.core as mx
from ..op import Op

class ExpandDims(Op):
    def __init__(self, inputs: str, outputs: str, axis: int = 0) -> None:
        super().__init__(inputs, outputs)
        self.axis = axis

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        return mx.expand_dims(data, axis=self.axis)

class Minmax(Op):
    def __init__(self, inputs: str, outputs: str) -> None:
        super().__init__(inputs, outputs)

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        return data.astype(mx.float32) / 255.0


class Normalize(Op):
    def __init__(self, inputs: str, outputs: str,
                 mean: Sequence[float] = (0.0, 0.0, 0.0),
                 std: Sequence[float] = (1.0, 1.0, 1.0),
                 max_pixel_value: float = 255.0) -> None:
        super().__init__(inputs, outputs)
        self.mean = mx.array(mean, dtype=mx.float32)
        self.std = mx.array(std, dtype=mx.float32)
        self.max_pixel_value = max_pixel_value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        x = data.astype(mx.float32) / self.max_pixel_value
        return (x - self.mean) / self.std


class PadIfNeeded(Op):
    def __init__(self, inputs: str, outputs: str,
                 min_height: int, min_width: int, value: float = 0.0) -> None:
        super().__init__(inputs, outputs)
        self.min_height = min_height
        self.min_width = min_width
        self.value = value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        x = np.array(data)
        if x.ndim == 4:
            batch, h, w, c = x.shape
        else:
            h, w, c = x.shape
            batch = None
        pad_h = max(0, self.min_height - h)
        pad_w = max(0, self.min_width - w)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_cfg = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)) if batch is not None else \
                       ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            x = np.pad(x, pad_cfg, mode="constant", constant_values=self.value)
        return mx.array(x)


class RandomCrop(Op):
    def __init__(self, inputs: str, outputs: str, height: int, width: int) -> None:
        super().__init__(inputs, outputs)
        self.height = height
        self.width = width

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        x = np.array(data)
        if x.ndim == 4:
            batch, h, w, c = x.shape
            result = np.empty((batch, self.height, self.width, c), dtype=x.dtype)
            for i in range(batch):
                top = np.random.randint(0, h - self.height + 1)
                left = np.random.randint(0, w - self.width + 1)
                result[i] = x[i, top:top + self.height, left:left + self.width, :]
        else:
            h, w, c = x.shape
            top = np.random.randint(0, h - self.height + 1)
            left = np.random.randint(0, w - self.width + 1)
            result = x[top:top + self.height, left:left + self.width, :]
        return mx.array(result)


class HorizontalFlip(Op):
    def __init__(self, inputs: str, outputs: str, prob: float = 0.5) -> None:
        super().__init__(inputs, outputs)
        self.prob = prob

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        x = np.array(data)
        if np.random.rand() < self.prob:
            if x.ndim == 4:
                x = x[:, :, ::-1, :]
            else:
                x = x[:, ::-1, :]
        return mx.array(x)


class CoarseDropout(Op):
    def __init__(self, inputs: str, outputs: str, max_holes: int = 1,
                 max_height: int = 8, max_width: int = 8, fill_value: float = 0.0) -> None:
        super().__init__(inputs, outputs)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.fill_value = fill_value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        x = np.array(data)
        if x.ndim == 4:
            batch, h, w, c = x.shape
            for i in range(batch):
                for _ in range(self.max_holes):
                    hole_h = np.random.randint(1, self.max_height + 1)
                    hole_w = np.random.randint(1, self.max_width + 1)
                    top = np.random.randint(0, max(1, h - hole_h + 1))
                    left = np.random.randint(0, max(1, w - hole_w + 1))
                    x[i, top:top + hole_h, left:left + hole_w, :] = self.fill_value
        else:
            h, w, c = x.shape
            for _ in range(self.max_holes):
                hole_h = np.random.randint(1, self.max_height + 1)
                hole_w = np.random.randint(1, self.max_width + 1)
                top = np.random.randint(0, max(1, h - hole_h + 1))
                left = np.random.randint(0, max(1, w - hole_w + 1))
                x[top:top + hole_h, left:left + hole_w, :] = self.fill_value
        return mx.array(x)


class Onehot(Op):
    def __init__(self, inputs: str, outputs: str, num_classes: int,
                 label_smoothing: float = 0.0) -> None:
        super().__init__(inputs, outputs)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if not isinstance(data, mx.array):
            data = mx.array(data)
        y = np.array(data).astype(int)
        oh = np.eye(self.num_classes, dtype=np.float32)[y]
        if self.label_smoothing:
            smooth = self.label_smoothing / self.num_classes
            oh = np.where(oh == 1.0, 1.0 - self.label_smoothing + smooth, smooth)
        return mx.array(oh)


class Sometimes(Op):
    def __init__(self, op: Op, prob: float = 0.5) -> None:
        super().__init__(op.inputs, op.outputs)
        self.op = op
        self.prob = prob

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Any:
        if np.random.rand() < self.prob:
            return self.op.forward(data, state)
        return data

