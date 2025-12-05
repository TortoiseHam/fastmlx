"""Lambda operation for custom transformations."""

from __future__ import annotations

from typing import Any, Callable, MutableMapping, Optional, Sequence, Union

import mlx.core as mx

from .op import Op


class LambdaOp(Op):
    """Apply a custom function to data.

    Args:
        fn: A function to apply to the data.
        inputs: Input key(s) for the data.
        outputs: Output key(s) for the transformed data.

    Example:
        >>> # Double all values
        >>> op = LambdaOp(lambda x: x * 2, inputs="x", outputs="x")
        >>> # Custom normalization
        >>> op = LambdaOp(lambda x: (x - x.mean()) / x.std(), inputs="x", outputs="x_norm")
    """

    def __init__(
        self,
        fn: Callable,
        inputs: Union[str, Sequence[str]],
        outputs: Union[str, Sequence[str]]
    ) -> None:
        super().__init__(inputs, outputs)
        self.fn = fn

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Any:
        return self.fn(data)


class Delete(Op):
    """Delete keys from the data dictionary.

    This is useful for removing intermediate values that are no longer needed.

    Args:
        keys: Key(s) to delete from the batch.

    Example:
        >>> op = Delete(keys=["intermediate_result", "temp_data"])
    """

    def __init__(self, keys: Union[str, Sequence[str]]) -> None:
        if isinstance(keys, str):
            keys = [keys]
        super().__init__(inputs=list(keys), outputs=[])
        self.keys = list(keys)

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> None:
        # The deletion happens at the network level, not in forward
        return None


class RemoveIf(Op):
    """Conditionally remove samples from a batch.

    Samples are removed if the condition function returns True.

    Args:
        fn: A function that takes data and returns True if the sample should be removed.
        inputs: Input key(s) to pass to the condition function.

    Example:
        >>> # Remove samples where label is -1
        >>> op = RemoveIf(lambda y: y == -1, inputs="y")
        >>> # Remove samples with NaN values
        >>> op = RemoveIf(lambda x: mx.any(mx.isnan(x)), inputs="x")
    """

    def __init__(
        self,
        fn: Callable,
        inputs: Union[str, Sequence[str]]
    ) -> None:
        super().__init__(inputs, outputs=[])
        self.fn = fn

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> bool:
        # Returns True if sample should be removed
        return self.fn(data)


class Reshape(Op):
    """Reshape array data to a new shape.

    Args:
        inputs: Input key for the data.
        outputs: Output key for the reshaped data.
        shape: Target shape. Use -1 for inferred dimension.

    Example:
        >>> # Flatten to 1D
        >>> op = Reshape(inputs="x", outputs="x", shape=(-1,))
        >>> # Reshape to specific size
        >>> op = Reshape(inputs="x", outputs="x", shape=(3, 32, 32))
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        shape: Sequence[int]
    ) -> None:
        super().__init__(inputs, outputs)
        self.shape = tuple(shape)

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        return data.reshape(self.shape)


class Cast(Op):
    """Cast array data to a different dtype.

    Args:
        inputs: Input key for the data.
        outputs: Output key for the cast data.
        dtype: Target data type (e.g., mx.float32, mx.int32).

    Example:
        >>> op = Cast(inputs="x", outputs="x", dtype=mx.float32)
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        dtype: mx.Dtype
    ) -> None:
        super().__init__(inputs, outputs)
        self.dtype = dtype

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        return data.astype(self.dtype)


class Squeeze(Op):
    """Remove single-dimensional entries from array shape.

    Args:
        inputs: Input key for the data.
        outputs: Output key for the squeezed data.
        axis: Axis to squeeze. If None, squeeze all single dimensions.

    Example:
        >>> op = Squeeze(inputs="x", outputs="x", axis=0)
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        axis: Optional[int] = None
    ) -> None:
        super().__init__(inputs, outputs)
        self.axis = axis

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        if self.axis is not None:
            return mx.squeeze(data, axis=self.axis)
        return mx.squeeze(data)


class Clip(Op):
    """Clip array values to a range.

    Args:
        inputs: Input key for the data.
        outputs: Output key for the clipped data.
        a_min: Minimum value.
        a_max: Maximum value.

    Example:
        >>> op = Clip(inputs="x", outputs="x", a_min=0.0, a_max=1.0)
    """

    def __init__(
        self,
        inputs: str,
        outputs: str,
        a_min: Optional[float] = None,
        a_max: Optional[float] = None
    ) -> None:
        super().__init__(inputs, outputs)
        self.a_min = a_min
        self.a_max = a_max

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> mx.array:
        return mx.clip(data, self.a_min, self.a_max)
