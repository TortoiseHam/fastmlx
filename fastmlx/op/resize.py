"""Resize augmentation operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Tuple

import mlx.core as mx
import numpy as np

from .op import Op


class Resize(Op):
    """Resize images to a target size.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the resized image.
        height: Target height.
        width: Target width.
        interpolation: Interpolation method ('nearest', 'bilinear').
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        height: int,
        width: int,
        interpolation: str = "bilinear"
    ) -> None:
        super().__init__(inputs, outputs)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize a single image using simple interpolation."""
        h, w = img.shape[:2]
        new_h, new_w = self.height, self.width

        # Create coordinate grids
        y = np.linspace(0, h - 1, new_h)
        x = np.linspace(0, w - 1, new_w)

        if self.interpolation == "nearest":
            yi = np.round(y).astype(int)
            xi = np.round(x).astype(int)
            yi = np.clip(yi, 0, h - 1)
            xi = np.clip(xi, 0, w - 1)
            return img[yi[:, None], xi[None, :]]
        else:  # bilinear
            y0 = np.floor(y).astype(int)
            x0 = np.floor(x).astype(int)
            y1 = np.minimum(y0 + 1, h - 1)
            x1 = np.minimum(x0 + 1, w - 1)

            fy = (y - y0)[:, None]
            fx = (x - x0)[None, :]

            # Bilinear interpolation
            result = (
                img[y0[:, None], x0[None, :]] * (1 - fy) * (1 - fx) +
                img[y0[:, None], x1[None, :]] * (1 - fy) * fx +
                img[y1[:, None], x0[None, :]] * fy * (1 - fx) +
                img[y1[:, None], x1[None, :]] * fy * fx
            )
            return result

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        x = np.array(data).astype(np.float32)

        if x.ndim == 4:
            # Batch
            result = []
            for img in x:
                result.append(self._resize_image(img))
            x = np.stack(result)
        else:
            x = self._resize_image(x)

        return mx.array(x)


class RandomResizedCrop(Op):
    """Crop a random portion of the image and resize to target size.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the cropped image.
        height: Target height.
        width: Target width.
        scale: Range of size of the origin size cropped.
        ratio: Range of aspect ratio of the origin aspect ratio cropped.
        prob: Probability of applying the crop.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        height: int,
        width: int,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.333),
        prob: float = 1.0
    ) -> None:
        super().__init__(inputs, outputs)
        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio
        self.prob = prob

    def _get_params(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Get crop parameters."""
        area = h * w

        for _ in range(10):
            target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = np.exp(np.random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1])))

            crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < crop_w <= w and 0 < crop_h <= h:
                top = np.random.randint(0, h - crop_h + 1)
                left = np.random.randint(0, w - crop_w + 1)
                return top, left, crop_h, crop_w

        # Fallback to center crop
        if w / h < self.ratio[0]:
            crop_w = w
            crop_h = int(round(w / self.ratio[0]))
        elif w / h > self.ratio[1]:
            crop_h = h
            crop_w = int(round(h * self.ratio[1]))
        else:
            crop_h, crop_w = h, w

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return top, left, crop_h, crop_w

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize using bilinear interpolation."""
        h, w = img.shape[:2]
        new_h, new_w = self.height, self.width

        y = np.linspace(0, h - 1, new_h)
        x = np.linspace(0, w - 1, new_w)

        y0 = np.floor(y).astype(int)
        x0 = np.floor(x).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)

        fy = (y - y0)[:, None]
        fx = (x - x0)[None, :]

        result = (
            img[y0[:, None], x0[None, :]] * (1 - fy) * (1 - fx) +
            img[y0[:, None], x1[None, :]] * (1 - fy) * fx +
            img[y1[:, None], x0[None, :]] * fy * (1 - fx) +
            img[y1[:, None], x1[None, :]] * fy * fx
        )
        return result

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            # Just resize without crop
            return Resize(self.inputs, self.outputs, self.height, self.width).forward(data, state)

        x = np.array(data).astype(np.float32)

        if x.ndim == 4:
            result = []
            for img in x:
                h, w = img.shape[:2]
                top, left, crop_h, crop_w = self._get_params(h, w)
                cropped = img[top:top + crop_h, left:left + crop_w]
                result.append(self._resize_image(cropped))
            x = np.stack(result)
        else:
            h, w = x.shape[:2]
            top, left, crop_h, crop_w = self._get_params(h, w)
            cropped = x[top:top + crop_h, left:left + crop_w]
            x = self._resize_image(cropped)

        return mx.array(x)


class CenterCrop(Op):
    """Crop the center of the image.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the cropped image.
        height: Target crop height.
        width: Target crop width.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        height: int,
        width: int
    ) -> None:
        super().__init__(inputs, outputs)
        self.height = height
        self.width = width

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        x = np.array(data)

        if x.ndim == 4:
            # Batch
            b, h, w, c = x.shape
        else:
            h, w = x.shape[:2]

        # Calculate crop coordinates
        top = (h - self.height) // 2
        left = (w - self.width) // 2

        if x.ndim == 4:
            x = x[:, top:top + self.height, left:left + self.width, :]
        else:
            x = x[top:top + self.height, left:left + self.width]

        return mx.array(x)
