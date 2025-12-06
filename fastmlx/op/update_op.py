"""Model update operation with gradient computation and optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, MutableMapping, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .op import Op

if TYPE_CHECKING:
    from ..backend.amp import GradScaler



class UpdateOp(Op):
    """Compute gradients and update model parameters.

    This op computes gradients of a loss with respect to model parameters
    and updates the model using the provided optimizer.

    Args:
        model: The model to update. Must have an `optimizer` attribute.
        loss_name: Key in batch containing the loss value to backpropagate.
        inputs: Input key(s) for the model forward pass.
        outputs: Output key(s) from the model (predictions).
        loss_fn: Optional custom loss function. If None, uses the loss value
                 directly from batch[loss_name]. If provided, should take
                 (predictions, targets) and return a scalar loss.
        accumulation_steps: Number of steps to accumulate gradients before
                           updating. Default is 1 (no accumulation).
        max_grad_norm: Maximum gradient norm for clipping. None for no clipping.
        compile: Whether to compile the step function for performance.

    Example:
        >>> # Using with a separate loss op (recommended)
        >>> network = Network(ops=[
        ...     ModelOp(model=model, inputs="x", outputs="y_pred"),
        ...     CrossEntropyLoss(inputs=("y_pred", "y"), outputs="ce"),
        ...     UpdateOp(model=model, loss_name="ce", inputs="x", outputs="y_pred")
        ... ])

        >>> # With gradient accumulation for larger effective batch size
        >>> UpdateOp(model=model, loss_name="ce", inputs="x", outputs="y_pred",
        ...          accumulation_steps=4)

        >>> # With gradient clipping
        >>> UpdateOp(model=model, loss_name="ce", inputs="x", outputs="y_pred",
        ...          max_grad_norm=1.0)

        >>> # With mixed precision (set by Estimator)
        >>> estimator = Estimator(..., mixed_precision=True)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_name: str = "loss",
        inputs: Optional[Union[str, List[str]]] = None,
        outputs: Optional[Union[str, List[str]]] = None,
        loss_fn: Optional[Callable] = None,
        accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        compile: bool = True
    ) -> None:
        # For backward compatibility, accept both old and new API
        super().__init__(inputs=inputs or [], outputs=outputs or [])
        self.model: nn.Module = model
        self.loss_name = loss_name
        self.loss_fn = loss_fn
        self.accumulation_steps = max(1, accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.compile = compile

        # Gradient accumulation state
        self._accumulated_grads: Optional[dict] = None
        self._accumulation_count: int = 0

        # Compilation state
        self._state: List[mx.array] = []
        self._step_fn: Optional[Callable] = None
        self._initialized = False

        # AMP support - set by Estimator if mixed_precision is enabled
        self.grad_scaler: Optional["GradScaler"] = None

    def _initialize(self) -> None:
        """Initialize the compiled step function."""
        if self._initialized:
            return

        if not hasattr(self.model, 'optimizer'):
            raise ValueError(
                "UpdateOp requires model to have an 'optimizer' attribute. "
                "Set model.optimizer = optim.SGD(...) before training."
            )

        self._state = [self.model.state, self.model.optimizer.state, mx.random.state]
        self._initialized = True

    def _compute_gradients(
        self,
        batch: MutableMapping[str, Any]
    ) -> tuple[mx.array, dict]:
        """Compute gradients of loss with respect to model parameters.

        Returns:
            Tuple of (loss_value, gradients_dict)
        """
        if self.loss_fn is not None:
            # Use custom loss function - need to do forward pass
            input_keys = self.inputs if isinstance(self.inputs, list) else [self.inputs]
            # output_keys not needed for custom loss forward pass

            # Get input data
            if len(input_keys) == 1:
                x = batch.get(input_keys[0])
            else:
                x = tuple(batch.get(k) for k in input_keys)

            # Get target data for loss
            y = batch.get("y")  # Standard target key

            def loss_wrapper(model: nn.Module) -> mx.array:
                pred = model(x) if not isinstance(x, tuple) else model(*x)
                loss = self.loss_fn(pred, y)
                return mx.mean(loss) if loss.ndim > 0 else loss

            loss_grad_fn = nn.value_and_grad(self.model, loss_wrapper)
            loss_val, grads = loss_grad_fn(self.model)

        else:
            # Use pre-computed loss from batch
            loss_val = batch.get(self.loss_name)
            if loss_val is None:
                raise ValueError(
                    f"UpdateOp: Loss key '{self.loss_name}' not found in batch. "
                    f"Available keys: {list(batch.keys())}. "
                    f"Make sure a loss op runs before UpdateOp."
                )

            # Ensure loss is a scalar
            if isinstance(loss_val, mx.array) and loss_val.ndim > 0:
                loss_val = mx.mean(loss_val)

            # Get input for gradient computation
            input_keys = self.inputs if isinstance(self.inputs, list) else [self.inputs]
            if input_keys and input_keys[0]:
                x = batch.get(input_keys[0])
            else:
                x = batch.get("x")  # Fallback to standard key

            y = batch.get("y")

            if x is None:
                raise ValueError(
                    f"UpdateOp: Input data not found. Looked for keys {input_keys} and 'x'. "
                    f"Available keys: {list(batch.keys())}"
                )

            # Create loss function that uses the model's forward pass
            def loss_wrapper(model: nn.Module) -> mx.array:
                pred = model(x)
                # Recompute the loss to get gradients
                # We need the prediction to compute gradients through the model
                if y is not None:
                    # Use cross entropy as default if we have targets
                    loss = nn.losses.cross_entropy(pred, y)
                    return mx.mean(loss)
                else:
                    # If no targets, use MSE with stored loss as target
                    # This is a fallback and may not work well
                    return loss_val

            loss_grad_fn = nn.value_and_grad(self.model, loss_wrapper)
            loss_val, grads = loss_grad_fn(self.model)

        return loss_val, grads

    def _clip_gradients(self, grads: dict) -> dict:
        """Clip gradients by global norm."""
        if self.max_grad_norm is None:
            return grads

        # Compute global norm
        total_norm_sq = mx.array(0.0)
        for key, value in grads.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, mx.array):
                        total_norm_sq = total_norm_sq + mx.sum(subvalue ** 2)
            elif isinstance(value, mx.array):
                total_norm_sq = total_norm_sq + mx.sum(value ** 2)

        total_norm = mx.sqrt(total_norm_sq)
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        clip_coef = mx.minimum(clip_coef, mx.array(1.0))

        # Apply clipping
        def clip_value(v):
            if isinstance(v, mx.array):
                return v * clip_coef
            elif isinstance(v, dict):
                return {k: clip_value(val) for k, val in v.items()}
            return v

        return {k: clip_value(v) for k, v in grads.items()}

    def _accumulate_gradients(self, grads: dict) -> None:
        """Accumulate gradients for gradient accumulation."""
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

    def forward(self, data: mx.array, state: MutableMapping[str, Any]) -> None:
        """Compute gradients and update model parameters.

        Args:
            data: Unused (we get data from state['batch'])
            state: Training state containing 'batch' and 'mode'
        """
        # Only update during training
        if state.get("mode") != "train":
            return None

        batch = state.get("batch", {})
        if not batch:
            return None

        # Initialize on first call
        self._initialize()

        # Compute gradients
        loss_val, grads = self._compute_gradients(batch)

        # Handle gradient scaling for mixed precision
        if self.grad_scaler is not None:
            # Unscale gradients before clipping and update
            grads = self.grad_scaler.unscale(grads)

        # Clip gradients if configured
        if self.max_grad_norm is not None:
            grads = self._clip_gradients(grads)

        # Handle gradient accumulation
        if self.accumulation_steps > 1:
            self._accumulate_gradients(grads)

            if self._should_update():
                # Apply accumulated gradients (with grad scaler if enabled)
                if self.grad_scaler is not None:
                    stepped = self.grad_scaler.step(
                        self.model.optimizer,
                        self.model,
                        self._accumulated_grads
                    )
                    if stepped:
                        self.grad_scaler.update()
                else:
                    self.model.optimizer.update(self.model, self._accumulated_grads)
                mx.eval(self.model.state, self.model.optimizer.state)
                self._reset_accumulation()
        else:
            # Standard update (with grad scaler if enabled)
            if self.grad_scaler is not None:
                stepped = self.grad_scaler.step(
                    self.model.optimizer,
                    self.model,
                    grads
                )
                if stepped:
                    self.grad_scaler.update()
            else:
                self.model.optimizer.update(self.model, grads)
            mx.eval(loss_val, self.model.state, self.model.optimizer.state)

        # Store loss in batch for logging
        batch[self.loss_name] = loss_val

        return None
