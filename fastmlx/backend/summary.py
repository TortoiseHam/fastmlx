"""Model summary utilities for inspecting neural network architectures."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the total number of parameters in a model.

    Args:
        model: MLX model to count parameters for.
        trainable_only: If True, only count trainable parameters.

    Returns:
        Total number of parameters.

    Example:
        >>> model = nn.Linear(100, 10)
        >>> count_parameters(model)
        1010  # 100*10 weights + 10 biases
    """
    total = 0

    def count_in_dict(d: Dict[str, Any]) -> int:
        count = 0
        for key, value in d.items():
            if isinstance(value, mx.array):
                count += value.size
            elif isinstance(value, dict):
                count += count_in_dict(value)
        return count

    params = model.parameters()
    total = count_in_dict(params)

    return total


def model_summary(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    batch_size: int = 1,
    show_trainable: bool = True,
    print_output: bool = True
) -> Dict[str, Any]:
    """Generate a summary of the model architecture.

    Args:
        model: MLX model to summarize.
        input_shape: Optional input shape (without batch dimension) for
                    calculating output shapes. If None, shapes won't be shown.
        batch_size: Batch size for shape calculation.
        show_trainable: Whether to show trainable parameter count.
        print_output: Whether to print the summary to console.

    Returns:
        Dictionary containing summary information.

    Example:
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10)
        ... )
        >>> summary = model_summary(model, input_shape=(784,))
    """
    summary_info: Dict[str, Any] = {
        "model_name": model.__class__.__name__,
        "layers": [],
        "total_params": 0,
        "trainable_params": 0,
        "non_trainable_params": 0,
    }

    # Count parameters
    total_params = count_parameters(model)
    summary_info["total_params"] = total_params
    summary_info["trainable_params"] = total_params  # MLX: all params are trainable by default
    summary_info["non_trainable_params"] = 0

    # Collect layer information
    layers = _collect_layers(model)
    summary_info["layers"] = layers

    if print_output:
        _print_summary(summary_info, input_shape, batch_size)

    return summary_info


def _collect_layers(
    module: nn.Module,
    prefix: str = ""
) -> List[Dict[str, Any]]:
    """Recursively collect layer information from a module."""
    layers = []

    # Get the module's own parameters
    own_params = {}
    if hasattr(module, "__dict__"):
        for name, value in module.__dict__.items():
            if isinstance(value, mx.array):
                own_params[name] = value

    # Calculate parameter count for this module
    param_count = sum(p.size for p in own_params.values())

    if param_count > 0 or not hasattr(module, "layers"):
        layer_info = {
            "name": prefix or module.__class__.__name__,
            "type": module.__class__.__name__,
            "params": param_count,
            "param_shapes": {k: v.shape for k, v in own_params.items()},
        }
        layers.append(layer_info)

    # Recursively process child modules
    if hasattr(module, "layers"):
        for i, layer in enumerate(module.layers):
            child_prefix = f"{prefix}.layers[{i}]" if prefix else f"layers[{i}]"
            layers.extend(_collect_layers(layer, child_prefix))
    elif hasattr(module, "__dict__"):
        for name, child in module.__dict__.items():
            if isinstance(child, nn.Module):
                child_prefix = f"{prefix}.{name}" if prefix else name
                layers.extend(_collect_layers(child, child_prefix))

    return layers


def _print_summary(
    summary_info: Dict[str, Any],
    input_shape: Optional[Tuple[int, ...]],
    batch_size: int
) -> None:
    """Print the model summary to console."""
    line_width = 80
    col_widths = [40, 20, 15]

    print("=" * line_width)
    print(f"Model: {summary_info['model_name']}")
    print("=" * line_width)

    # Header
    headers = ["Layer (type)", "Param Shape", "Param #"]
    header_line = ""
    for header, width in zip(headers, col_widths):
        header_line += header.ljust(width)
    print(header_line)
    print("-" * line_width)

    # Layer rows
    for layer in summary_info["layers"]:
        name = layer["name"]
        layer_type = layer["type"]
        params = layer["params"]

        # Format name with type
        name_str = f"{name} ({layer_type})"
        if len(name_str) > col_widths[0] - 2:
            name_str = name_str[: col_widths[0] - 5] + "..."

        # Format shape
        shapes = layer.get("param_shapes", {})
        if shapes:
            shape_strs = [f"{k}: {v}" for k, v in shapes.items()]
            shape_str = ", ".join(shape_strs)
            if len(shape_str) > col_widths[1] - 2:
                shape_str = shape_str[: col_widths[1] - 5] + "..."
        else:
            shape_str = "-"

        # Format param count
        param_str = f"{params:,}"

        row = name_str.ljust(col_widths[0])
        row += shape_str.ljust(col_widths[1])
        row += param_str.rjust(col_widths[2])
        print(row)

    print("=" * line_width)

    # Summary
    total_params = summary_info["total_params"]
    trainable_params = summary_info["trainable_params"]

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {summary_info['non_trainable_params']:,}")

    # Memory estimate (assuming float32)
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"Estimated memory (float32): {memory_mb:.2f} MB")

    print("=" * line_width)


def layer_info(layer: nn.Module) -> Dict[str, Any]:
    """Get detailed information about a single layer.

    Args:
        layer: MLX layer to inspect.

    Returns:
        Dictionary with layer information.
    """
    info = {
        "type": layer.__class__.__name__,
        "params": {},
        "total_params": 0,
    }

    # Get parameters
    if hasattr(layer, "weight"):
        info["params"]["weight"] = {
            "shape": layer.weight.shape,
            "dtype": str(layer.weight.dtype),
            "size": layer.weight.size,
        }
        info["total_params"] += layer.weight.size

    if hasattr(layer, "bias") and layer.bias is not None:
        info["params"]["bias"] = {
            "shape": layer.bias.shape,
            "dtype": str(layer.bias.dtype),
            "size": layer.bias.size,
        }
        info["total_params"] += layer.bias.size

    # Layer-specific info
    if isinstance(layer, nn.Linear):
        info["input_features"] = layer.weight.shape[1]
        info["output_features"] = layer.weight.shape[0]

    elif isinstance(layer, nn.Conv2d):
        info["in_channels"] = layer.weight.shape[-1]
        info["out_channels"] = layer.weight.shape[0]
        info["kernel_size"] = layer.weight.shape[1:3]

    elif isinstance(layer, nn.Embedding):
        info["num_embeddings"] = layer.weight.shape[0]
        info["embedding_dim"] = layer.weight.shape[1]

    elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm)):
        info["normalized_shape"] = layer.weight.shape

    return info


def compare_models(
    model1: nn.Module,
    model2: nn.Module,
    name1: str = "Model 1",
    name2: str = "Model 2"
) -> Dict[str, Any]:
    """Compare two models by parameter count and structure.

    Args:
        model1: First model to compare.
        model2: Second model to compare.
        name1: Name for first model.
        name2: Name for second model.

    Returns:
        Dictionary with comparison results.
    """
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)

    comparison = {
        name1: {
            "type": model1.__class__.__name__,
            "total_params": params1,
            "memory_mb": (params1 * 4) / (1024 * 1024),
        },
        name2: {
            "type": model2.__class__.__name__,
            "total_params": params2,
            "memory_mb": (params2 * 4) / (1024 * 1024),
        },
        "difference": {
            "params": params2 - params1,
            "ratio": params2 / params1 if params1 > 0 else float("inf"),
        },
    }

    print(f"\nModel Comparison:")
    print(f"{'='*50}")
    print(f"{name1}: {params1:,} params ({comparison[name1]['memory_mb']:.2f} MB)")
    print(f"{name2}: {params2:,} params ({comparison[name2]['memory_mb']:.2f} MB)")
    print(f"Difference: {params2 - params1:+,} params ({comparison['difference']['ratio']:.2f}x)")
    print(f"{'='*50}")

    return comparison
