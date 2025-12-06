"""Backend utilities for MLX operations."""

from .amp import (
    AMPConfig,
    GradScaler,
    amp_forward,
    cast_to_dtype,
    estimate_memory_savings,
    get_model_dtype,
)
from .device import (
    device_info,
    get_default_device,
    get_device,
    is_gpu_available,
    set_device,
    synchronize,
)
from .dtype import (
    bfloat16,
    bool_,
    float16,
    float32,
    get_dtype,
    int32,
    int64,
    to_dtype,
    uint8,
)
from .memory import (
    clear_cache,
    memory_info,
    peak_memory,
    reset_peak_memory,
)
from .seed import (
    get_seed,
    set_seed,
)
from .summary import (
    compare_models,
    count_parameters,
    layer_info,
    model_summary,
)

__all__ = [
    # Device utilities
    "get_device",
    "set_device",
    "get_default_device",
    "is_gpu_available",
    "device_info",
    "synchronize",
    # Memory utilities
    "memory_info",
    "clear_cache",
    "peak_memory",
    "reset_peak_memory",
    # Data type utilities
    "get_dtype",
    "to_dtype",
    "float16",
    "float32",
    "bfloat16",
    "int32",
    "int64",
    "uint8",
    "bool_",
    # Seed utilities
    "set_seed",
    "get_seed",
    # Model summary utilities
    "count_parameters",
    "model_summary",
    "layer_info",
    "compare_models",
    # AMP utilities
    "AMPConfig",
    "GradScaler",
    "cast_to_dtype",
    "get_model_dtype",
    "amp_forward",
    "estimate_memory_savings",
]
