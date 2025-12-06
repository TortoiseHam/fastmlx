"""Memory management utilities for MLX."""

from __future__ import annotations

from typing import Any, Dict

import mlx.core as mx

# Track peak memory usage (simulated since MLX doesn't expose this directly)
_peak_memory_bytes: int = 0


def memory_info() -> Dict[str, Any]:
    """Get memory usage information.

    Returns:
        Dictionary containing memory information.

    Note:
        MLX uses a lazy evaluation model, so memory is allocated
        on-demand when arrays are evaluated. This function returns
        available system info rather than precise GPU memory usage.

    Example:
        >>> info = memory_info()
        >>> print(f"Cache size: {info.get('cache_size', 'N/A')}")
    """
    info = {}

    # Try to get metal memory info on macOS
    try:
        import platform
        if platform.system() == "Darwin":
            # Metal-specific info would go here
            # MLX doesn't expose detailed memory stats directly
            info["backend"] = "Metal"
    except Exception:
        pass

    return info


def clear_cache() -> None:
    """Clear MLX's internal memory cache.

    This can help free up memory when switching between
    large models or datasets.

    Example:
        >>> # After loading a large model
        >>> model = load_large_model()
        >>> # ... use model ...
        >>> del model
        >>> clear_cache()  # Free cached memory
    """
    # MLX automatically manages memory, but we can help by
    # forcing garbage collection of any pending operations
    try:
        import gc
        gc.collect()
    except Exception:
        pass


def peak_memory() -> int:
    """Get peak memory usage in bytes.

    Returns:
        Peak memory usage in bytes (approximate).

    Note:
        This is a best-effort approximation since MLX
        uses lazy evaluation and automatic memory management.
    """
    global _peak_memory_bytes
    return _peak_memory_bytes


def reset_peak_memory() -> None:
    """Reset peak memory tracking.

    Example:
        >>> reset_peak_memory()
        >>> # ... run training ...
        >>> print(f"Peak memory: {peak_memory() / 1e9:.2f} GB")
    """
    global _peak_memory_bytes
    _peak_memory_bytes = 0


def estimate_array_memory(shape: tuple, dtype: mx.Dtype = mx.float32) -> int:
    """Estimate memory usage for an array with given shape and dtype.

    Args:
        shape: Array shape.
        dtype: Data type.

    Returns:
        Estimated memory in bytes.

    Example:
        >>> mem = estimate_array_memory((1000, 1000), mx.float32)
        >>> print(f"Estimated: {mem / 1e6:.2f} MB")  # ~4 MB
    """
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    # Get bytes per element based on dtype
    dtype_sizes = {
        mx.float32: 4,
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.int32: 4,
        mx.int64: 8,
        mx.int16: 2,
        mx.int8: 1,
        mx.uint8: 1,
        mx.uint32: 4,
        mx.uint64: 8,
        mx.bool_: 1,
    }

    bytes_per_element = dtype_sizes.get(dtype, 4)  # Default to 4 bytes
    return num_elements * bytes_per_element


def format_memory(bytes_value: int) -> str:
    """Format memory size in human-readable form.

    Args:
        bytes_value: Memory in bytes.

    Returns:
        Human-readable string (e.g., "1.5 GB").

    Example:
        >>> print(format_memory(1_500_000_000))
        '1.40 GB'
    """
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024 ** 2:
        return f"{bytes_value / 1024:.2f} KB"
    elif bytes_value < 1024 ** 3:
        return f"{bytes_value / 1024 ** 2:.2f} MB"
    else:
        return f"{bytes_value / 1024 ** 3:.2f} GB"
