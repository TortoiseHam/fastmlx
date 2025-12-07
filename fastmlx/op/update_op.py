"""Model update operation configuration for gradient-based optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, MutableMapping, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .op import Op

if TYPE_CHECKING:
    from ..backend.amp import GradScaler


class UpdateOp(Op):
    """Configuration for gradient computation and parameter updates.

    UpdateOp specifies which model to update, which loss to use, and optional
    settings like gradient clipping and accumulation. The actual gradient
    computation is handled by the Network class, which composes all ops into
    an efficient single forward pass using MLX's functional autodiff.

    Args:
        model: The model to update. Must have an `optimizer` attribute.
        loss_name: Key in batch containing the loss value to backpropagate.
        accumulation_steps: Number of steps to accumulate gradients before
                           updating. Default is 1 (no accumulation).
        max_grad_norm: Maximum gradient norm for clipping. None for no clipping.

    Example:
        >>> # Standard usage - Network handles efficient gradient computation
        >>> network = Network([
        ...     ModelOp(model=model, inputs="x", outputs="y_pred"),
        ...     CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        ...     UpdateOp(model=model, loss_name="ce")
        ... ])

        >>> # With gradient accumulation for larger effective batch size
        >>> UpdateOp(model=model, loss_name="ce", accumulation_steps=4)

        >>> # With gradient clipping
        >>> UpdateOp(model=model, loss_name="ce", max_grad_norm=1.0)

    Note:
        The Network class automatically composes ModelOp, LossOp, and UpdateOp
        into a single traced function for `nn.value_and_grad()`. This avoids
        the inefficiency of recomputing the forward pass for gradient computation,
        which is necessary due to MLX's functional autodiff (unlike PyTorch's
        tape-based autodiff where tensors carry computation history).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_name: str = "loss",
        accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        super().__init__(inputs=[], outputs=[])
        self.model: nn.Module = model
        self.loss_name = loss_name
        self.accumulation_steps = max(1, accumulation_steps)
        self.max_grad_norm = max_grad_norm

        # Gradient accumulation state
        self._accumulated_grads: Optional[dict] = None
        self._accumulation_count: int = 0

        # AMP support - set by Estimator if mixed_precision is enabled
        self.grad_scaler: Optional["GradScaler"] = None

    def _accumulate_gradients(self, grads: dict) -> None:
        """Accumulate gradients for gradient accumulation.

        Called by Network when accumulation_steps > 1.
        """
        if self._accumulated_grads is None:
            # First accumulation - just store (scaled by accumulation steps)
            scale = 1.0 / self.accumulation_steps

            def scale_value(v):
                if isinstance(v, mx.array):
                    return v * scale
                elif isinstance(v, dict):
                    return {k: scale_value(val) for k, val in v.items()}
                return v

            self._accumulated_grads = {k: scale_value(v) for k, v in grads.items()}
        else:
            # Add to existing gradients (scaled)
            scale = 1.0 / self.accumulation_steps

            def add_grads(acc, new):
                if isinstance(acc, mx.array) and isinstance(new, mx.array):
                    return acc + new * scale
                elif isinstance(acc, dict) and isinstance(new, dict):
                    return {k: add_grads(acc[k], new[k]) for k in acc.keys()}
                return acc

            self._accumulated_grads = {
                k: add_grads(self._accumulated_grads[k], grads[k])
                for k in self._accumulated_grads.keys()
            }

        self._accumulation_count += 1

    def _should_update(self) -> bool:
        """Check if we should perform the parameter update."""
        return self._accumulation_count >= self.accumulation_steps

    def _reset_accumulation(self) -> None:
        """Reset gradient accumulation state."""
        self._accumulated_grads = None
        self._accumulation_count = 0

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> None:
        """No-op. Gradient computation is handled by Network.

        This method exists for API compatibility but does nothing.
        The Network class intercepts UpdateOp and handles gradient
        computation efficiently via op composition.
        """
        return None
