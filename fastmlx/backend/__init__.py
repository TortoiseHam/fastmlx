"""Backend utilities for MLX operations."""

from .device import (
    get_device,
    set_device,
    get_default_device,
    is_gpu_available,
    device_info,
    synchronize,
)

from .memory import (
    memory_info,
    clear_cache,
    peak_memory,
    reset_peak_memory,
)

from .dtype import (
    get_dtype,
    to_dtype,
    float16,
    float32,
    bfloat16,
    int32,
    int64,
    uint8,
    bool_,
)

from .seed import (
    set_seed,
    get_seed,
)

from .summary import (
    count_parameters,
    model_summary,
    layer_info,
    compare_models,
)

from .amp import (
    AMPConfig,
    GradScaler,
    cast_to_dtype,
    get_model_dtype,
    amp_forward,
    estimate_memory_savings,
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
