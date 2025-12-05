"""Gaussian blur augmentation operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Tuple, Union

import numpy as np
import mlx.core as mx

from .op import Op


class GaussianBlur(Op):
    """Apply Gaussian blur to images.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the blurred image.
        blur_limit: Maximum kernel size for the blur.
                   Can be int or tuple (min, max). Must be odd.
        sigma_limit: Range for sigma. Can be float or tuple (min, max).
        prob: Probability of applying the blur.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        blur_limit: Union[int, Tuple[int, int]] = (3, 7),
        sigma_limit: Union[float, Tuple[float, float]] = (0.1, 2.0),
        prob: float = 0.5
    ) -> None:
        super().__init__(inputs, outputs)
        if isinstance(blur_limit, int):
            self.blur_limit = (3, blur_limit)
        else:
            self.blur_limit = blur_limit
        if isinstance(sigma_limit, (int, float)):
            self.sigma_limit = (0.1, float(sigma_limit))
        else:
            self.sigma_limit = sigma_limit
        self.prob = prob

    def _gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Generate a 2D Gaussian kernel."""
        x = np.arange(size) - (size - 1) / 2
        kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / kernel_2d.sum()

    def _apply_blur(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply blur using convolution."""
        try:
            from scipy.ndimage import convolve
            # Apply to each channel separately
            if img.ndim == 3:
                result = np.zeros_like(img)
                for c in range(img.shape[-1]):
                    result[..., c] = convolve(img[..., c], kernel, mode='reflect')
                return result
            else:
                return convolve(img, kernel, mode='reflect')
        except ImportError:
            # Simple fallback - just return original if scipy not available
            return img

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data).astype(np.float32)

        # Random kernel size (must be odd)
        ksize = np.random.randint(self.blur_limit[0] // 2, self.blur_limit[1] // 2 + 1) * 2 + 1
        sigma = np.random.uniform(self.sigma_limit[0], self.sigma_limit[1])

        kernel = self._gaussian_kernel(ksize, sigma)

        if x.ndim == 4:
            # Batch
            result = []
            for img in x:
                result.append(self._apply_blur(img, kernel))
            x = np.stack(result)
        else:
            x = self._apply_blur(x, kernel)

        return mx.array(x)
