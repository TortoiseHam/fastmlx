"""Cutout (random erasing) augmentation operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Tuple, Union

import numpy as np
import mlx.core as mx

from .op import Op


class Cutout(Op):
    """Apply Cutout (random erasing) augmentation.

    Randomly selects a rectangle region and erases its pixels.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the augmented image.
        num_holes: Number of holes to cut.
        max_h_size: Maximum height of the hole.
        max_w_size: Maximum width of the hole.
        fill_value: Value to fill the hole with.
        prob: Probability of applying cutout.

    Reference:
        DeVries & Taylor, "Improved Regularization of Convolutional Neural Networks
        with Cutout", arXiv 2017.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        num_holes: int = 1,
        max_h_size: int = 8,
        max_w_size: int = 8,
        fill_value: float = 0.0,
        prob: float = 0.5
    ) -> None:
        super().__init__(inputs, outputs)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.prob = prob

    def _apply_cutout(self, img: np.ndarray) -> np.ndarray:
        """Apply cutout to a single image."""
        h, w = img.shape[:2]
        img = img.copy()

        for _ in range(self.num_holes):
            # Random center
            y = np.random.randint(h)
            x = np.random.randint(w)

            # Random size
            hole_h = np.random.randint(1, self.max_h_size + 1)
            hole_w = np.random.randint(1, self.max_w_size + 1)

            # Calculate bounds
            y1 = max(0, y - hole_h // 2)
            y2 = min(h, y + hole_h // 2)
            x1 = max(0, x - hole_w // 2)
            x2 = min(w, x + hole_w // 2)

            # Fill hole
            img[y1:y2, x1:x2] = self.fill_value

        return img

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data).astype(np.float32)

        if x.ndim == 4:
            result = []
            for img in x:
                result.append(self._apply_cutout(img))
            x = np.stack(result)
        else:
            x = self._apply_cutout(x)

        return mx.array(x)


class GridMask(Op):
    """Apply GridMask augmentation.

    Creates a grid-like mask and applies it to the image.

    Args:
        inputs: Input key for the image.
        outputs: Output key for the augmented image.
        ratio: Ratio of the grid to be masked.
        d_range: Range of grid size.
        fill_value: Value to fill the masked regions.
        prob: Probability of applying GridMask.

    Reference:
        Chen et al., "GridMask Data Augmentation", arXiv 2020.
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        ratio: float = 0.5,
        d_range: Tuple[int, int] = (96, 224),
        fill_value: float = 0.0,
        prob: float = 0.5
    ) -> None:
        super().__init__(inputs, outputs)
        self.ratio = ratio
        self.d_range = d_range
        self.fill_value = fill_value
        self.prob = prob

    def _apply_gridmask(self, img: np.ndarray) -> np.ndarray:
        """Apply GridMask to a single image."""
        h, w = img.shape[:2]
        img = img.copy()

        # Random grid size
        d = np.random.randint(self.d_range[0], self.d_range[1])

        # Grid parameters
        l = int(d * self.ratio)  # masked region size

        # Random offset
        delta_y = np.random.randint(d)
        delta_x = np.random.randint(d)

        # Create mask
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(-1, h // d + 2):
            y = i * d + delta_y
            for j in range(-1, w // d + 2):
                x = j * d + delta_x
                y1 = max(0, y)
                y2 = min(h, y + l)
                x1 = max(0, x)
                x2 = min(w, x + l)
                if y1 < y2 and x1 < x2:
                    mask[y1:y2, x1:x2] = 0

        # Apply mask
        if img.ndim == 3:
            mask = mask[:, :, np.newaxis]
        img = img * mask + self.fill_value * (1 - mask)

        return img

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data).astype(np.float32)

        if x.ndim == 4:
            result = []
            for img in x:
                result.append(self._apply_gridmask(img))
            x = np.stack(result)
        else:
            x = self._apply_gridmask(x)

        return mx.array(x)


class MixUp(Op):
    """Apply MixUp augmentation between batch samples.

    Note: This op works on batches and mixes samples within the batch.
    Should be applied after batching.

    Args:
        inputs: Tuple of (image_key, label_key).
        outputs: Tuple of (mixed_image_key, mixed_label_key).
        alpha: Beta distribution parameter for mixing coefficient.
        prob: Probability of applying MixUp.

    Reference:
        Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
    """

    def __init__(
        self,
        inputs: Tuple[str, str],
        outputs: Tuple[str, str],
        alpha: float = 0.2,
        prob: float = 0.5
    ) -> None:
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.prob = prob

    def forward(self, data: Tuple[mx.array, mx.array], state: MutableMapping[str, Any]) -> Tuple[mx.array, mx.array]:
        if np.random.rand() >= self.prob:
            return data

        x, y = data
        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.float32)

        batch_size = x.shape[0]

        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)

        # Random permutation
        indices = np.random.permutation(batch_size)

        # Mix
        mixed_x = lam * x + (1 - lam) * x[indices]
        mixed_y = lam * y + (1 - lam) * y[indices]

        return mx.array(mixed_x), mx.array(mixed_y)
