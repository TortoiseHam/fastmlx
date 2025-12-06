"""Automatic Mixed Precision (AMP) utilities for MLX.

On Apple Silicon, mixed precision training primarily helps with:
1. Memory savings (fit larger models/batches)
2. Modest speedup on certain operations

Unlike NVIDIA Tensor Cores, Apple Silicon doesn't have dedicated
low-precision matrix units, so speedups are less dramatic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

# Supported dtypes for mixed precision
HALF_DTYPES = (mx.float16, mx.bfloat16)
FULL_DTYPES = (mx.float32, mx.float64)


def cast_to_dtype(model: nn.Module, dtype: mx.Dtype) -> nn.Module:
    """Cast all model parameters to the specified dtype.

    Args:
        model: MLX model to cast.
        dtype: Target dtype (e.g., mx.float16, mx.bfloat16).

    Returns:
        The model with parameters cast to the new dtype.

    Example:
        >>> model = nn.Linear(100, 10)
        >>> model = cast_to_dtype(model, mx.float16)
    """
    def cast_params(params: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in params.items():
            if isinstance(value, mx.array):
                if value.dtype in FULL_DTYPES or value.dtype in HALF_DTYPES:
                    result[key] = value.astype(dtype)
                else:
                    result[key] = value  # Keep int types as-is
            elif isinstance(value, dict):
                result[key] = cast_params(value)
            else:
                result[key] = value
        return result

    # Get current parameters, cast them, and update
    params = model.parameters()
    casted_params = cast_params(params)
    model.update(casted_params)

    return model


def get_model_dtype(model: nn.Module) -> Optional[mx.Dtype]:
    """Get the dtype of a model's parameters.

    Args:
        model: MLX model to inspect.

    Returns:
        The dtype of the first parameter found, or None if no parameters.
    """
    params = model.parameters()

    def find_dtype(d: Dict[str, Any]) -> Optional[mx.Dtype]:
        for value in d.values():
            if isinstance(value, mx.array):
                return value.dtype
            elif isinstance(value, dict):
                result = find_dtype(value)
                if result is not None:
                    return result
        return None

    return find_dtype(params)


class AMPConfig:
    """Configuration for Automatic Mixed Precision training.

    Args:
        enabled: Whether AMP is enabled.
        dtype: The reduced precision dtype to use (float16 or bfloat16).
        loss_scale: Static loss scaling factor. If None, no loss scaling.
        grad_scale: Whether to scale gradients (useful for float16 stability).

    Example:
        >>> config = AMPConfig(enabled=True, dtype=mx.float16)
        >>> # Use with Estimator
        >>> estimator = Estimator(..., amp_config=config)
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: mx.Dtype = mx.float16,
        loss_scale: Optional[float] = None,
        grad_scale: bool = False
    ) -> None:
        self.enabled = enabled
        self.dtype = dtype
        self.loss_scale = loss_scale
        self.grad_scale = grad_scale

        if dtype not in HALF_DTYPES:
            raise ValueError(
                f"AMP dtype must be float16 or bfloat16, got {dtype}"
            )

    def __repr__(self) -> str:
        return (
            f"AMPConfig(enabled={self.enabled}, dtype={self.dtype}, "
            f"loss_scale={self.loss_scale})"
        )


def amp_forward(
    model: nn.Module,
    x: mx.array,
    dtype: mx.Dtype = mx.float16
) -> mx.array:
    """Run forward pass with automatic dtype casting.

    Casts input to reduced precision, runs forward, and returns output
    in the same precision as input.

    Args:
        model: MLX model to run.
        x: Input tensor.
        dtype: Reduced precision dtype.

    Returns:
        Model output (in original input dtype).
    """
    input_dtype = x.dtype

    # Cast input to reduced precision
    x_half = x.astype(dtype)

    # Forward pass in reduced precision
    out = model(x_half)

    # Cast back to original precision
    return out.astype(input_dtype)


def scaled_loss(loss: mx.array, scale: float) -> mx.array:
    """Scale loss for numerical stability with float16.

    Args:
        loss: The computed loss value.
        scale: Scaling factor.

    Returns:
        Scaled loss.
    """
    return loss * scale


def unscale_grads(grads: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """Unscale gradients after backward pass.

    Args:
        grads: Dictionary of gradients.
        scale: The scale factor used for loss scaling.

    Returns:
        Unscaled gradients.
    """
    inv_scale = 1.0 / scale

    def unscale(d: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in d.items():
            if isinstance(value, mx.array):
                result[key] = value * inv_scale
            elif isinstance(value, dict):
                result[key] = unscale(value)
            else:
                result[key] = value
        return result

    return unscale(grads)


def check_gradients_finite(grads: Dict[str, Any]) -> bool:
    """Check if all gradients are finite (no NaN or Inf).

    Args:
        grads: Dictionary of gradients.

    Returns:
        True if all gradients are finite, False otherwise.
    """
    def check_finite(d: Dict[str, Any]) -> bool:
        for value in d.values():
            if isinstance(value, mx.array):
                if not mx.all(mx.isfinite(value)):
                    return False
            elif isinstance(value, dict):
                if not check_finite(value):
                    return False
        return True

    return check_finite(grads)


class GradScaler:
    """Gradient scaler for mixed precision training.

    Scales loss to prevent underflow in float16, then unscales
    gradients before optimizer step.

    Args:
        init_scale: Initial scale factor.
        growth_factor: Factor to increase scale when no overflow.
        backoff_factor: Factor to decrease scale on overflow.
        growth_interval: Steps between scale increases.

    Example:
        >>> scaler = GradScaler(init_scale=65536.0)
        >>> # In training loop:
        >>> scaled_loss = scaler.scale(loss)
        >>> grads = compute_gradients(scaled_loss)
        >>> grads = scaler.unscale(grads)
        >>> if scaler.step(optimizer, model, grads):
        ...     print("Updated successfully")
        >>> scaler.update()
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ) -> None:
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0

    @property
    def scale_value(self) -> float:
        """Current scale value."""
        return self._scale

    def scale(self, loss: mx.array) -> mx.array:
        """Scale the loss value.

        Args:
            loss: Unscaled loss.

        Returns:
            Scaled loss.
        """
        return loss * self._scale

    def unscale(self, grads: Dict[str, Any]) -> Dict[str, Any]:
        """Unscale gradients.

        Args:
            grads: Scaled gradients.

        Returns:
            Unscaled gradients.
        """
        return unscale_grads(grads, self._scale)

    def step(
        self,
        optimizer,
        model: nn.Module,
        grads: Dict[str, Any]
    ) -> bool:
        """Perform optimizer step if gradients are finite.

        Args:
            optimizer: The optimizer to step.
            model: The model to update.
            grads: Unscaled gradients.

        Returns:
            True if step was performed, False if skipped due to overflow.
        """
        if check_gradients_finite(grads):
            optimizer.update(model, grads)
            self._growth_tracker += 1
            return True
        else:
            # Overflow detected - skip this step
            return False

    def update(self) -> None:
        """Update scale factor based on overflow history."""
        if self._growth_tracker >= self._growth_interval:
            self._scale *= self._growth_factor
            self._growth_tracker = 0


def estimate_memory_savings(model: nn.Module) -> Dict[str, float]:
    """Estimate memory savings from using float16.

    Args:
        model: Model to analyze.

    Returns:
        Dictionary with memory estimates.
    """
    from .summary import count_parameters

    num_params = count_parameters(model)

    fp32_memory_mb = (num_params * 4) / (1024 * 1024)  # 4 bytes per float32
    fp16_memory_mb = (num_params * 2) / (1024 * 1024)  # 2 bytes per float16

    return {
        "num_parameters": num_params,
        "fp32_memory_mb": fp32_memory_mb,
        "fp16_memory_mb": fp16_memory_mb,
        "savings_mb": fp32_memory_mb - fp16_memory_mb,
        "savings_percent": ((fp32_memory_mb - fp16_memory_mb) / fp32_memory_mb) * 100,
    }
