"""Utilities for visualizing batches of images using Plotly."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import mlx.core as mx
import numpy as np
from plotly.graph_objects import Figure, Image
from plotly.subplots import make_subplots


def _to_numpy(arr: Union[mx.array, np.ndarray]) -> np.ndarray:
    """Convert an MLX array or numpy array into numpy."""
    if isinstance(arr, mx.array):
        return np.array(arr)
    return np.asarray(arr)


class ImageDisplay:
    """Display a single image."""

    def __init__(self, image: Union[mx.array, np.ndarray], title: Optional[str] = None) -> None:
        img = _to_numpy(image)
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[-1]:
            img = np.moveaxis(img, 0, -1)
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        self.image = img
        self.title = title or ""

    def prepare(self, fig: Figure, row: int, col: int) -> Figure:
        fig.add_trace(Image(z=self.image), row=row, col=col)
        return fig


class BatchDisplay:
    """Display a batch of images vertically."""

    def __init__(
        self,
        image: Optional[Union[mx.array, np.ndarray, Sequence[Union[mx.array, np.ndarray]]]] = None,
        text: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
    ) -> None:
        imgs: List[ImageDisplay] = []
        if image is not None:
            img_np = _to_numpy(image)
            if img_np.ndim == 3:
                img_np = np.expand_dims(img_np, 0)
            if img_np.shape[1] in (1, 3) and img_np.shape[1] != img_np.shape[-1]:
                img_np = np.moveaxis(img_np, 1, -1)
            for i in range(img_np.shape[0]):
                label = None
                if text is not None:
                    if len(text) == img_np.shape[0]:
                        label = str(text[i])
                    elif len(text) == 1:
                        label = str(text[0])
                imgs.append(ImageDisplay(img_np[i], title=label))
        self.images = imgs
        self.title = title or ""
        self.batch_size = len(self.images)

    def prepare(self) -> Figure:
        fig = make_subplots(rows=max(1, self.batch_size), cols=1,
                            subplot_titles=[img.title or "" for img in self.images] or [self.title])
        for idx, img in enumerate(self.images, start=1):
            img.prepare(fig, row=idx, col=1)
        height = 300 * max(1, self.batch_size)
        fig.update_layout(width=300, height=height, showlegend=False)
        return fig


class GridDisplay:
    """Display multiple :class:`BatchDisplay` objects side by side."""

    def __init__(self, columns: Sequence[BatchDisplay], title: str | None = None) -> None:
        if not columns:
            raise ValueError("At least one column must be provided")
        batch_size = columns[0].batch_size
        for c in columns:
            if c.batch_size != batch_size:
                raise ValueError("All columns must have the same batch size")
        self.columns = list(columns)
        self.batch_size = batch_size
        self.title = title or "grid"

    def prepare(self) -> Figure:
        fig = make_subplots(rows=max(1, self.batch_size),
                            cols=len(self.columns),
                            column_titles=[c.title for c in self.columns])
        for col_idx, column in enumerate(self.columns, start=1):
            for row_idx, img in enumerate(column.images, start=1):
                img.prepare(fig, row=row_idx, col=col_idx)
        width = 300 * len(self.columns)
        height = 300 * max(1, self.batch_size)
        fig.update_layout(width=width, height=height, showlegend=False)
        return fig

