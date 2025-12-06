"""Device management utilities for MLX."""

from __future__ import annotations

import platform
from typing import Any, Dict, Optional

import mlx.core as mx


def get_device() -> mx.Device:
    """Get the current default device.

    Returns:
        The current default MLX device.

    Example:
        >>> device = get_device()
        >>> print(device)  # e.g., Device(gpu, 0)
    """
    return mx.default_device()


def set_device(device: str) -> None:
    """Set the default device.

    Args:
        device: Device type - "gpu", "cpu", or device string.

    Example:
        >>> set_device("gpu")  # Use GPU
        >>> set_device("cpu")  # Use CPU
    """
    if device.lower() == "gpu":
        mx.set_default_device(mx.gpu)
    elif device.lower() == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        raise ValueError(f"Unknown device: {device}. Use 'gpu' or 'cpu'.")


def get_default_device() -> str:
    """Get a string representation of the default device.

    Returns:
        Device type string ("gpu" or "cpu").
    """
    device = mx.default_device()
    return "gpu" if device == mx.gpu else "cpu"


def is_gpu_available() -> bool:
    """Check if GPU is available.

    Returns:
        True if GPU is available, False otherwise.

    Note:
        On macOS, this checks for Metal support.
        On Linux, this checks for CUDA support (MLX 0.30.0+).
    """
    try:
        # Try to create a small array on GPU
        mx.set_default_device(mx.gpu)
        test = mx.zeros((1,))
        mx.eval(test)
        return True
    except Exception:
        return False


def device_info() -> Dict[str, Any]:
    """Get detailed device information.

    Returns:
        Dictionary containing device information.

    Example:
        >>> info = device_info()
        >>> print(info['default_device'])
        >>> print(info['platform'])
    """
    info = {
        "default_device": str(mx.default_device()),
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "mlx_version": mx.__version__ if hasattr(mx, "__version__") else "unknown",
    }

    # Platform-specific info
    if platform.system() == "Darwin":
        info["backend"] = "Metal"
        info["apple_silicon"] = platform.machine() == "arm64"
    elif platform.system() == "Linux":
        info["backend"] = "CUDA"  # MLX 0.30.0+ supports CUDA on Linux

    return info


def synchronize() -> None:
    """Synchronize all pending operations.

    This ensures all queued operations are completed before returning.
    Useful for accurate timing measurements.

    Example:
        >>> import time
        >>> x = mx.random.normal((1000, 1000))
        >>> y = mx.matmul(x, x)
        >>> synchronize()  # Wait for matmul to complete
        >>> # Now y is guaranteed to be computed
    """
    mx.eval(mx.zeros((1,)))
