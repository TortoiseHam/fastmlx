"""Color jitter augmentation operation."""

from __future__ import annotations

from typing import Any, MutableMapping

import mlx.core as mx
import numpy as np

from .op import Op


class ColorJitter(Op):
    """Randomly change brightness, contrast, saturation, and hue.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the adjusted image.
        brightness: Range for brightness adjustment.
        contrast: Range for contrast adjustment.
        saturation: Range for saturation adjustment.
        hue: Range for hue adjustment (in degrees, will be converted to 0-1 range).
        prob: Probability of applying adjustments.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        prob: float = 0.5
    ) -> None:
        super().__init__(inputs, outputs)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        s = np.where(maxc != 0, (maxc - minc) / maxc, 0)

        rc = np.where(maxc != minc, (maxc - r) / (maxc - minc), 0)
        gc = np.where(maxc != minc, (maxc - g) / (maxc - minc), 0)
        bc = np.where(maxc != minc, (maxc - b) / (maxc - minc), 0)

        h = np.where(r == maxc, bc - gc, np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
        h = (h / 6.0) % 1.0

        return np.stack([h, s, v], axis=-1)

    def _hsv_to_rgb(self, hsv: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB."""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype(np.int32)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        rgb = np.zeros(hsv.shape, dtype=np.float32)
        mask = (i == 0)
        rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=-1)
        mask = (i == 1)
        rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=-1)
        mask = (i == 2)
        rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=-1)
        mask = (i == 3)
        rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=-1)
        mask = (i == 4)
        rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=-1)
        mask = (i == 5)
        rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=-1)

        return rgb

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data).astype(np.float32)
        is_batch = x.ndim == 4

        # Determine max value and normalize
        max_val = 255.0 if x.max() > 1.0 else 1.0
        x = x / max_val

        # Random order of transforms
        transforms = list(range(4))
        np.random.shuffle(transforms)

        for t in transforms:
            if t == 0 and self.brightness > 0:
                # Brightness
                factor = np.random.uniform(1 - self.brightness, 1 + self.brightness)
                x = x * factor
            elif t == 1 and self.contrast > 0:
                # Contrast
                factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
                mean = x.mean(axis=(-3, -2, -1) if is_batch else (-3, -2), keepdims=True)
                x = (x - mean) * factor + mean
            elif t == 2 and self.saturation > 0:
                # Saturation (convert to HSV)
                if x.shape[-1] == 3:  # Only for RGB
                    factor = np.random.uniform(1 - self.saturation, 1 + self.saturation)
                    if is_batch:
                        hsv = np.stack([self._rgb_to_hsv(img) for img in x])
                        hsv[..., 1] = hsv[..., 1] * factor
                        x = np.stack([self._hsv_to_rgb(h) for h in hsv])
                    else:
                        hsv = self._rgb_to_hsv(x)
                        hsv[..., 1] = hsv[..., 1] * factor
                        x = self._hsv_to_rgb(hsv)
            elif t == 3 and self.hue > 0:
                # Hue shift
                if x.shape[-1] == 3:  # Only for RGB
                    shift = np.random.uniform(-self.hue, self.hue)
                    if is_batch:
                        hsv = np.stack([self._rgb_to_hsv(img) for img in x])
                        hsv[..., 0] = (hsv[..., 0] + shift) % 1.0
                        x = np.stack([self._hsv_to_rgb(h) for h in hsv])
                    else:
                        hsv = self._rgb_to_hsv(x)
                        hsv[..., 0] = (hsv[..., 0] + shift) % 1.0
                        x = self._hsv_to_rgb(hsv)

        # Clip and restore scale
        x = np.clip(x, 0, 1) * max_val

        return mx.array(x)


class ToGray(Op):
    """Convert RGB images to grayscale.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the grayscale image.
        prob: Probability of applying conversion.
        keep_channels: If True, output has 3 identical channels.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        prob: float = 1.0,
        keep_channels: bool = True
    ) -> None:
        super().__init__(inputs, outputs)
        self.prob = prob
        self.keep_channels = keep_channels

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data).astype(np.float32)

        # Only convert if RGB (3 channels)
        if x.shape[-1] != 3:
            return data

        # Standard grayscale weights
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray = np.sum(x * weights, axis=-1, keepdims=True)

        if self.keep_channels:
            gray = np.repeat(gray, 3, axis=-1)

        return mx.array(gray)
