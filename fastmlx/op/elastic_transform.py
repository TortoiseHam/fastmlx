"""Elastic Transform augmentation operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Optional, Tuple

import mlx.core as mx
import numpy as np

from .op import Op


class ElasticTransform(Op):
    """Apply elastic deformation to images.

    This transform applies random elastic deformations to images,
    commonly used for data augmentation in image segmentation tasks.

    Args:
        inputs: Input key for images.
        outputs: Output key for augmented images.
        alpha: Scaling factor for displacement field.
        sigma: Standard deviation of Gaussian filter for smoothing.
        alpha_affine: Scaling factor for affine transformation.
        prob: Probability of applying the transform.
        border_mode: How to handle borders. Options: 'constant', 'reflect', 'wrap'.
        fill_value: Fill value for constant border mode.

    Example:
        >>> op = ElasticTransform(inputs="x", outputs="x", alpha=100, sigma=10)
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        alpha: float = 100.0,
        sigma: float = 10.0,
        alpha_affine: float = 10.0,
        prob: float = 0.5,
        border_mode: str = "constant",
        fill_value: float = 0
    ) -> None:
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.prob = prob
        self.border_mode = border_mode
        self.fill_value = fill_value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data)

        if x.ndim == 4:
            # Batch of images
            for i in range(x.shape[0]):
                x[i] = self._elastic_transform(x[i])
        elif x.ndim == 3:
            x = self._elastic_transform(x)

        return mx.array(x)

    def _elastic_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply elastic transform to a single image."""
        h, w = img.shape[:2]

        # Random displacement fields
        dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
        dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32)

        # Smooth with Gaussian filter
        dx = self._gaussian_filter(dx, self.sigma) * self.alpha
        dy = self._gaussian_filter(dy, self.sigma) * self.alpha

        # Create coordinate grids
        y, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Apply displacement
        map_x = (x_grid + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        # Remap image
        return self._remap(img, map_x, map_y)

    def _gaussian_filter(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian filter for smoothing."""
        if sigma <= 0:
            return x

        # Create Gaussian kernel
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1

        kernel_1d = np.exp(-np.arange(-(size // 2), size // 2 + 1) ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Apply separable filter
        result = np.apply_along_axis(
            lambda m: np.convolve(m, kernel_1d, mode='same'), axis=0, arr=x
        )
        result = np.apply_along_axis(
            lambda m: np.convolve(m, kernel_1d, mode='same'), axis=1, arr=result
        )

        return result

    def _remap(
        self,
        img: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray
    ) -> np.ndarray:
        """Remap image using bilinear interpolation."""
        h, w = img.shape[:2]
        # Channel count (unused, kept for potential future use)
        _ = img.shape[2] if img.ndim == 3 else 1

        # Clip coordinates
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)

        # Get integer and fractional parts
        x0 = np.floor(map_x).astype(int)
        y0 = np.floor(map_y).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)

        fx = map_x - x0
        fy = map_y - y0

        # Bilinear interpolation
        if img.ndim == 3:
            fx = fx[:, :, np.newaxis]
            fy = fy[:, :, np.newaxis]

            result = (
                img[y0, x0] * (1 - fx) * (1 - fy) +
                img[y0, x1] * fx * (1 - fy) +
                img[y1, x0] * (1 - fx) * fy +
                img[y1, x1] * fx * fy
            )
        else:
            result = (
                img[y0, x0] * (1 - fx) * (1 - fy) +
                img[y0, x1] * fx * (1 - fy) +
                img[y1, x0] * (1 - fx) * fy +
                img[y1, x1] * fx * fy
            )

        return result.astype(img.dtype)


class PerspectiveTransform(Op):
    """Apply random perspective transformation to images.

    Args:
        inputs: Input key for images.
        outputs: Output key for augmented images.
        scale: Range of perspective distortion scale.
        prob: Probability of applying the transform.
        fill_value: Fill value for areas outside the image.

    Example:
        >>> op = PerspectiveTransform(inputs="x", outputs="x", scale=(0.05, 0.1))
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        scale: Tuple[float, float] = (0.05, 0.1),
        prob: float = 0.5,
        fill_value: float = 0
    ) -> None:
        super().__init__(inputs, outputs)
        self.scale = scale
        self.prob = prob
        self.fill_value = fill_value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data)

        if x.ndim == 4:
            for i in range(x.shape[0]):
                x[i] = self._perspective_transform(x[i])
        elif x.ndim == 3:
            x = self._perspective_transform(x)

        return mx.array(x)

    def _perspective_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply perspective transform to a single image."""
        h, w = img.shape[:2]

        # Generate random perspective distortion
        scale = np.random.uniform(self.scale[0], self.scale[1])

        # Random offsets for corners
        offsets = np.random.uniform(-scale, scale, (4, 2))

        # Source points (corners)
        src = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        # Destination points (distorted corners)
        dst = src + offsets * np.array([[w, h]])

        # Compute perspective transform matrix
        matrix = self._get_perspective_matrix(src, dst)

        if matrix is None:
            return img

        # Apply transform
        return self._warp_perspective(img, matrix, (w, h))

    def _get_perspective_matrix(
        self,
        src: np.ndarray,
        dst: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute perspective transformation matrix."""
        # Build system of equations
        A = []
        b = []

        for i in range(4):
            x, y = src[i]
            u, v = dst[i]
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
            b.append(u)
            b.append(v)

        A = np.array(A)
        b = np.array(b)

        try:
            h = np.linalg.solve(A, b)
            h = np.append(h, 1).reshape(3, 3)
            return h
        except np.linalg.LinAlgError:
            return None

    def _warp_perspective(
        self,
        img: np.ndarray,
        matrix: np.ndarray,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """Warp image using perspective matrix."""
        w, h = output_size

        # Create output coordinate grid
        x_out, y_out = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones_like(x_out)
        coords = np.stack([x_out, y_out, ones], axis=-1)

        # Inverse transform
        try:
            inv_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return img

        # Apply inverse transform to get source coordinates
        src_coords = coords @ inv_matrix.T
        src_coords = src_coords[:, :, :2] / src_coords[:, :, 2:3]

        map_x = src_coords[:, :, 0].astype(np.float32)
        map_y = src_coords[:, :, 1].astype(np.float32)

        # Use simple nearest neighbor for now
        map_x = np.clip(np.round(map_x), 0, img.shape[1] - 1).astype(int)
        map_y = np.clip(np.round(map_y), 0, img.shape[0] - 1).astype(int)

        return img[map_y, map_x]


class ShearTransform(Op):
    """Apply random shear transformation to images.

    Args:
        inputs: Input key for images.
        outputs: Output key for augmented images.
        shear_range: Range of shear angle in degrees.
        prob: Probability of applying the transform.
        fill_value: Fill value for areas outside the image.

    Example:
        >>> op = ShearTransform(inputs="x", outputs="x", shear_range=(-20, 20))
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        shear_range: Tuple[float, float] = (-20, 20),
        prob: float = 0.5,
        fill_value: float = 0
    ) -> None:
        super().__init__(inputs, outputs)
        self.shear_range = shear_range
        self.prob = prob
        self.fill_value = fill_value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data)

        if x.ndim == 4:
            for i in range(x.shape[0]):
                x[i] = self._shear_transform(x[i])
        elif x.ndim == 3:
            x = self._shear_transform(x)

        return mx.array(x)

    def _shear_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply shear transform to a single image."""
        h, w = img.shape[:2]

        # Random shear angles
        shear_x = np.random.uniform(self.shear_range[0], self.shear_range[1])
        shear_y = np.random.uniform(self.shear_range[0], self.shear_range[1])

        # Convert to radians
        shear_x = np.tan(np.radians(shear_x))
        shear_y = np.tan(np.radians(shear_y))

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Center coordinates
        cx, cy = w / 2, h / 2
        x_centered = x_coords - cx
        y_centered = y_coords - cy

        # Apply shear
        new_x = x_centered + shear_x * y_centered + cx
        new_y = y_centered + shear_y * x_centered + cy

        # Clip to valid range
        new_x = np.clip(new_x, 0, w - 1).astype(int)
        new_y = np.clip(new_y, 0, h - 1).astype(int)

        # Remap
        result = np.full_like(img, self.fill_value)
        valid_mask = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h)
        result[y_coords[valid_mask], x_coords[valid_mask]] = img[new_y[valid_mask], new_x[valid_mask]]

        return result
