"""Data type utilities for MLX."""

from __future__ import annotations

from typing import Union

import mlx.core as mx

# Common dtype aliases
float16 = mx.float16
float32 = mx.float32
bfloat16 = mx.bfloat16
int32 = mx.int32
int64 = mx.int64
uint8 = mx.uint8
bool_ = mx.bool_


def get_dtype(name: str) -> mx.Dtype:
    """Get MLX dtype from string name.

    Args:
        name: Data type name (e.g., "float32", "float16", "int32").

    Returns:
        Corresponding MLX dtype.

    Raises:
        ValueError: If dtype name is not recognized.

    Example:
        >>> dtype = get_dtype("float16")
        >>> x = mx.zeros((10,), dtype=dtype)
    """
    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
        "int8": mx.int8,
        "int16": mx.int16,
        "int32": mx.int32,
        "int64": mx.int64,
        "uint8": mx.uint8,
        "uint16": mx.uint16,
        "uint32": mx.uint32,
        "uint64": mx.uint64,
        "bool": mx.bool_,
        "bool_": mx.bool_,
        # Common aliases
        "fp16": mx.float16,
        "fp32": mx.float32,
        "bf16": mx.bfloat16,
        "half": mx.float16,
        "float": mx.float32,
        "int": mx.int32,
    }

    name_lower = name.lower()
    if name_lower not in dtype_map:
        raise ValueError(
            f"Unknown dtype: {name}. "
            f"Available: {list(dtype_map.keys())}"
        )

    return dtype_map[name_lower]


def to_dtype(x: mx.array, dtype: Union[str, mx.Dtype]) -> mx.array:
    """Convert array to specified dtype.

    Args:
        x: Input array.
        dtype: Target dtype (string or mx.Dtype).

    Returns:
        Array converted to the target dtype.

    Example:
        >>> x = mx.array([1.0, 2.0, 3.0])
        >>> x_fp16 = to_dtype(x, "float16")
        >>> x_int = to_dtype(x, mx.int32)
    """
    if isinstance(dtype, str):
        dtype = get_dtype(dtype)
    return x.astype(dtype)


def dtype_info(dtype: Union[str, mx.Dtype]) -> dict:
    """Get information about a dtype.

    Args:
        dtype: Data type to get info for.

    Returns:
        Dictionary with dtype information.

    Example:
        >>> info = dtype_info("float16")
        >>> print(info['bits'])  # 16
    """
    if isinstance(dtype, str):
        dtype = get_dtype(dtype)

    info_map = {
        mx.float16: {"bits": 16, "category": "float", "name": "float16"},
        mx.float32: {"bits": 32, "category": "float", "name": "float32"},
        mx.bfloat16: {"bits": 16, "category": "float", "name": "bfloat16"},
        mx.int8: {"bits": 8, "category": "int", "name": "int8"},
        mx.int16: {"bits": 16, "category": "int", "name": "int16"},
        mx.int32: {"bits": 32, "category": "int", "name": "int32"},
        mx.int64: {"bits": 64, "category": "int", "name": "int64"},
        mx.uint8: {"bits": 8, "category": "uint", "name": "uint8"},
        mx.uint16: {"bits": 16, "category": "uint", "name": "uint16"},
        mx.uint32: {"bits": 32, "category": "uint", "name": "uint32"},
        mx.uint64: {"bits": 64, "category": "uint", "name": "uint64"},
        mx.bool_: {"bits": 8, "category": "bool", "name": "bool"},
    }

    return info_map.get(dtype, {"bits": 0, "category": "unknown", "name": str(dtype)})


def promote_types(dtype1: mx.Dtype, dtype2: mx.Dtype) -> mx.Dtype:
    """Get the promoted dtype for two dtypes.

    Args:
        dtype1: First dtype.
        dtype2: Second dtype.

    Returns:
        Promoted dtype that can represent both inputs.

    Example:
        >>> result = promote_types(mx.float16, mx.float32)
        >>> print(result)  # float32
    """
    # Create dummy arrays and let MLX handle promotion
    x1 = mx.zeros((1,), dtype=dtype1)
    x2 = mx.zeros((1,), dtype=dtype2)
    return (x1 + x2).dtype
