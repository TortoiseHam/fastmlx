"""Sentinel class for filtering samples from the pipeline."""

from __future__ import annotations


class FilteredData:
    """Sentinel returned from Op.forward() to indicate a sample should be dropped.

    When an Op returns FilteredData instead of processed data, the Pipeline
    will exclude that sample from the batch. This is useful for:
    - Removing samples that don't meet quality criteria
    - Implementing hard negative mining
    - Dynamic curriculum learning

    Args:
        replacement: If True, the Pipeline will attempt to fetch a replacement
            sample to maintain batch size. If False, the sample is simply dropped
            which may result in smaller batches. Default is True.

    Example:
        >>> class DropSmallImages(Op):
        ...     def __init__(self, min_size: int = 64):
        ...         super().__init__(inputs="x", outputs="x")
        ...         self.min_size = min_size
        ...
        ...     def forward(self, data, state):
        ...         if data.shape[0] < self.min_size or data.shape[1] < self.min_size:
        ...             return FilteredData()  # Drop this sample
        ...         return data

        >>> class HardNegativeMining(Op):
        ...     def __init__(self, threshold: float = 0.5):
        ...         super().__init__(inputs=["loss"], outputs=[])
        ...         self.threshold = threshold
        ...
        ...     def forward(self, data, state):
        ...         # Only keep samples with loss above threshold
        ...         if data < self.threshold:
        ...             return FilteredData(replacement=False)
        ...         return data
    """

    __slots__ = ("replacement",)

    def __init__(self, replacement: bool = True) -> None:
        self.replacement = replacement

    def __repr__(self) -> str:
        return f"FilteredData(replacement={self.replacement})"

    def __bool__(self) -> bool:
        # Always falsy so `if result:` checks work intuitively
        return False
