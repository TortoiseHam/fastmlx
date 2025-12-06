"""Random Erasing augmentation operation."""

from __future__ import annotations

from typing import Any, MutableMapping, Optional, Tuple

import mlx.core as mx
import numpy as np

from .op import Op


class RandomErasing(Op):
    """Randomly erase a rectangular region in an image.

    This is similar to Cutout but with random aspect ratios and
    the erased region is filled with random values or a constant.

    Args:
        inputs: Input key for images.
        outputs: Output key for augmented images.
        prob: Probability of applying the transform.
        scale: Range of proportion of erased area against image area.
        ratio: Range of aspect ratio of erased area.
        value: Fill value. Can be a float (constant), 'random' for random noise,
               or a tuple of 3 floats for RGB channels.

    Example:
        >>> op = RandomErasing(inputs="x", outputs="x", prob=0.5, scale=(0.02, 0.33))
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        prob: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Any = 0
    ) -> None:
        super().__init__(inputs, outputs)
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data)

        # Get image dimensions
        if x.ndim == 4:
            # Batch of images (N, H, W, C)
            n, h, w, c = x.shape
            for i in range(n):
                x[i] = self._erase_single(x[i])
        elif x.ndim == 3:
            # Single image (H, W, C)
            h, w, c = x.shape
            x = self._erase_single(x)
        else:
            return data  # Don't modify if not an image

        return mx.array(x)

    def _erase_single(self, img: np.ndarray) -> np.ndarray:
        """Apply random erasing to a single image."""
        h, w = img.shape[:2]
        c = img.shape[2] if img.ndim == 3 else 1
        area = h * w

        for _ in range(10):  # Try up to 10 times to find valid erase params
            target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = np.exp(
                np.random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1]))
            )

            erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(target_area / aspect_ratio)))

            if erase_h < h and erase_w < w:
                i = np.random.randint(0, h - erase_h + 1)
                j = np.random.randint(0, w - erase_w + 1)

                # Determine fill value
                if self.value == "random":
                    fill = np.random.uniform(0, 1, (erase_h, erase_w, c))
                elif isinstance(self.value, (tuple, list)):
                    fill = np.array(self.value).reshape(1, 1, -1)
                    fill = np.broadcast_to(fill, (erase_h, erase_w, c)).copy()
                else:
                    fill = np.full((erase_h, erase_w, c), self.value)

                img[i:i + erase_h, j:j + erase_w, :] = fill
                break

        return img


class GridDropout(Op):
    """Drop out grid cells randomly from an image.

    Similar to GridMask but simpler - drops out random grid cells.

    Args:
        inputs: Input key for images.
        outputs: Output key for augmented images.
        prob: Probability of applying the transform.
        ratio: Ratio of the grid cell to drop.
        grid_size: Size of the grid. If None, uses random size.
        fill_value: Value to fill dropped cells with.

    Example:
        >>> op = GridDropout(inputs="x", outputs="x", ratio=0.5, grid_size=8)
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        prob: float = 0.5,
        ratio: float = 0.5,
        grid_size: Optional[int] = None,
        fill_value: float = 0
    ) -> None:
        super().__init__(inputs, outputs)
        self.prob = prob
        self.ratio = ratio
        self.grid_size = grid_size
        self.fill_value = fill_value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data)

        if x.ndim == 4:
            for i in range(x.shape[0]):
                x[i] = self._apply_grid_dropout(x[i])
        elif x.ndim == 3:
            x = self._apply_grid_dropout(x)

        return mx.array(x)

    def _apply_grid_dropout(self, img: np.ndarray) -> np.ndarray:
        """Apply grid dropout to a single image."""
        h, w = img.shape[:2]

        # Determine grid size
        if self.grid_size is None:
            grid_size = np.random.randint(4, 16)
        else:
            grid_size = self.grid_size

        # Create mask
        mask = np.ones_like(img)

        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                if np.random.rand() < self.ratio:
                    i_end = min(i + grid_size, h)
                    j_end = min(j + grid_size, w)
                    mask[i:i_end, j:j_end, :] = 0

        # Apply mask
        img = img * mask + self.fill_value * (1 - mask)
        return img


class ChannelDropout(Op):
    """Randomly drop image channels.

    Args:
        inputs: Input key for images.
        outputs: Output key for augmented images.
        prob: Probability of applying the transform.
        channel_drop_range: Range for number of channels to drop (min, max).
        fill_value: Value to fill dropped channels with.

    Example:
        >>> op = ChannelDropout(inputs="x", outputs="x", channel_drop_range=(1, 2))
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        prob: float = 0.5,
        channel_drop_range: Tuple[int, int] = (1, 1),
        fill_value: float = 0
    ) -> None:
        super().__init__(inputs, outputs)
        self.prob = prob
        self.channel_drop_range = channel_drop_range
        self.fill_value = fill_value

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if np.random.rand() >= self.prob:
            return data

        x = np.array(data)

        if x.ndim < 3:
            return data

        c = x.shape[-1]
        if c <= 1:
            return data

        # Determine number of channels to drop
        min_drop, max_drop = self.channel_drop_range
        max_drop = min(max_drop, c - 1)  # Keep at least one channel
        num_drop = np.random.randint(min_drop, max_drop + 1)

        # Select channels to drop
        channels_to_drop = np.random.choice(c, num_drop, replace=False)

        # Apply dropout
        x[..., channels_to_drop] = self.fill_value

        return mx.array(x)
