"""A simple sequential network abstraction."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Set, Tuple

import mlx.core as mx
import mlx.nn as nn

from .op.op import Op


class NetworkError(Exception):
    """Exception raised for network execution errors."""
    pass


class Network:
    """Execute a sequence of :class:`~fastmlx.op.Op` objects.

    The Network class handles both inference and training. For training, it
    automatically composes all ops into a single differentiable function to
    avoid redundant forward passes (which is required for MLX's functional autodiff).

    Args:
        ops: Sequence of Op objects to execute in order.

    Example:
        >>> # This works efficiently - Network composes into single forward pass
        >>> network = Network([
        ...     ModelOp(model=model, inputs="x", outputs="y_pred"),
        ...     CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        ...     UpdateOp(model=model, loss_name="ce")
        ... ])
        >>> result = network.run(batch, state)

    MLX Autodiff Note:
        Unlike PyTorch where tensors carry computation history, MLX uses functional
        autodiff (like JAX). This means we can't compute a loss and then call
        .backward() on it. Instead, we must wrap the entire forward+loss computation
        in a function and pass it to nn.value_and_grad().

        The Network class handles this automatically by:
        1. Identifying which ops need to participate in gradient computation
        2. Composing them into a single traced function
        3. Running that function through nn.value_and_grad()
    """

    def __init__(self, ops: Sequence[Op]) -> None:
        self.ops: List[Op] = list(ops)
        self._compiled_train_fn: Optional[Callable] = None
        self._models: List[nn.Module] = []
        self._update_ops: List[Any] = []  # Will be UpdateOp instances

    def _analyze_ops(self) -> None:
        """Analyze ops to identify models and update ops for efficient training."""
        from .op.model_op import ModelOp
        from .op.update_op import UpdateOp

        self._models = []
        self._update_ops = []
        seen_models = set()

        for op in self.ops:
            if isinstance(op, ModelOp):
                model_id = id(op.model)
                if model_id not in seen_models:
                    self._models.append(op.model)
                    seen_models.add(model_id)
            elif isinstance(op, UpdateOp):
                self._update_ops.append(op)
                model_id = id(op.model)
                if model_id not in seen_models:
                    self._models.append(op.model)
                    seen_models.add(model_id)

    def _build_forward_loss_fn(
        self,
        batch: MutableMapping[str, Any],
        state: MutableMapping[str, Any]
    ) -> Tuple[Callable, List[str]]:
        """Build a function that computes forward pass and all losses.

        Returns:
            Tuple of (loss_function, list_of_loss_keys)
        """
        from .op.model_op import ModelOp
        from .op.update_op import UpdateOp

        # Identify loss keys from UpdateOps
        loss_keys = []
        for op in self._update_ops:
            if op.loss_name not in loss_keys:
                loss_keys.append(op.loss_name)

        current_mode = state.get("mode")

        def forward_and_loss(*models: nn.Module) -> mx.array:
            """Execute all ops and return combined loss."""
            # Create a mapping from model id to the passed-in model
            model_map = {id(m): models[i] for i, m in enumerate(self._models)}

            # Working store starts with batch data
            store = dict(batch)

            for op in self.ops:
                # Skip UpdateOp - we handle gradients separately
                if isinstance(op, UpdateOp):
                    continue

                # Check mode filtering
                if hasattr(op, "should_run") and not op.should_run(current_mode):
                    continue

                # Get inputs
                if len(op.inputs) == 1:
                    inp = store.get(op.inputs[0])
                else:
                    inp = [store.get(k) for k in op.inputs]

                # Execute op
                if isinstance(op, ModelOp):
                    # Use the model from our traced parameters
                    model = model_map.get(id(op.model), op.model)
                    is_training = current_mode == "train"
                    model.train(is_training)

                    if isinstance(inp, list) and len(op.inputs) > 1:
                        out = model(*inp)
                    else:
                        out = model(inp)
                else:
                    out = op.forward(inp, state)

                # Store outputs
                if op.outputs:
                    if len(op.outputs) == 1:
                        store[op.outputs[0]] = out
                    elif out is not None:
                        for k, v in zip(op.outputs, out):
                            store[k] = v

            # Collect and sum all losses
            total_loss = mx.array(0.0)
            for loss_key in loss_keys:
                loss_val = store.get(loss_key)
                if loss_val is not None:
                    total_loss = total_loss + loss_val

            # Store intermediate results for metrics (captured via closure)
            batch.update(store)

            return total_loss

        return forward_and_loss, loss_keys

    def _validate_inputs(
        self,
        op: Op,
        store: MutableMapping[str, Any],
        op_index: int
    ) -> None:
        """Validate that all input keys exist in the store."""
        for key in op.inputs:
            if key not in store:
                available = list(store.keys())
                raise NetworkError(
                    f"Op {op_index} ({op.__class__.__name__}) requires input '{key}' "
                    f"but it was not found. Available keys: {available}"
                )

    def run(
        self,
        batch: MutableMapping[str, Any],
        state: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Execute all ops on the batch.

        For training mode with UpdateOps, this automatically composes all ops
        into a single differentiable function for efficient gradient computation.

        Args:
            batch: Dictionary containing input data.
            state: Dictionary for sharing state between ops and traces.
                   Should contain 'mode' key for mode filtering.

        Returns:
            The modified batch dictionary with outputs added.
        """
        current_mode = state.get("mode")

        # Analyze ops if not done yet
        if not self._models:
            self._analyze_ops()

        # Training mode with UpdateOps - use composed function
        if current_mode == "train" and self._update_ops:
            return self._run_training(batch, state)

        # Inference/eval mode - run ops sequentially
        return self._run_inference(batch, state)

    def _run_inference(
        self,
        batch: MutableMapping[str, Any],
        state: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Run ops sequentially for inference/eval."""
        from .op.update_op import UpdateOp

        store = batch
        current_mode = state.get("mode")

        for i, op in enumerate(self.ops):
            # Skip UpdateOp in non-training mode
            if isinstance(op, UpdateOp):
                continue

            # Check mode filtering
            if hasattr(op, "should_run") and not op.should_run(current_mode):
                continue

            # Validate inputs
            self._validate_inputs(op, store, i)

            # Get inputs
            try:
                if len(op.inputs) == 1:
                    inp = store[op.inputs[0]]
                else:
                    inp = [store[k] for k in op.inputs]
            except KeyError as e:
                raise NetworkError(
                    f"Op {i} ({op.__class__.__name__}): Missing input key {e}"
                ) from e

            # Execute
            try:
                out = op.forward(inp, state)
            except Exception as e:
                raise NetworkError(
                    f"Op {i} ({op.__class__.__name__}): Forward pass failed: {e}"
                ) from e

            # Store outputs
            if op.outputs:
                if len(op.outputs) == 1:
                    store[op.outputs[0]] = out
                elif out is not None:
                    try:
                        for k, v in zip(op.outputs, out):
                            store[k] = v
                    except TypeError as e:
                        raise NetworkError(
                            f"Op {i} ({op.__class__.__name__}): Output not iterable."
                        ) from e

        return store

    def _run_training(
        self,
        batch: MutableMapping[str, Any],
        state: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Run training with composed forward+loss function.

        This method:
        1. Builds a single function containing all forward passes and loss computations
        2. Uses nn.value_and_grad to compute gradients in one forward pass
        3. Applies updates from all UpdateOps
        """
        from .op.update_op import UpdateOp

        state["batch"] = batch

        # Build the composed forward+loss function
        forward_loss_fn, loss_keys = self._build_forward_loss_fn(batch, state)

        # Create value_and_grad for all models
        # We need gradients w.r.t. all model parameters
        if len(self._models) == 1:
            # Single model case - simpler
            model = self._models[0]
            loss_and_grad = nn.value_and_grad(model, forward_loss_fn)
            total_loss, grads = loss_and_grad(model)

            # Apply gradients via UpdateOps
            for update_op in self._update_ops:
                if id(update_op.model) == id(model):
                    self._apply_update(update_op, grads, total_loss, state)

        else:
            # Multiple models - compute gradients for each
            # This is more complex; we compute gradients one model at a time
            for model in self._models:
                # Create a wrapper that only takes this model as differentiable
                def model_loss_fn(m: nn.Module) -> mx.array:
                    # Replace this model in the list
                    models_list = list(self._models)
                    idx = self._models.index(model)
                    models_list[idx] = m
                    return forward_loss_fn(*models_list)

                loss_and_grad = nn.value_and_grad(model, model_loss_fn)
                total_loss, grads = loss_and_grad(model)

                # Apply gradients for this model's UpdateOps
                for update_op in self._update_ops:
                    if id(update_op.model) == id(model):
                        self._apply_update(update_op, grads, total_loss, state)

        # Evaluate all model states
        for model in self._models:
            mx.eval(model.state)
            if hasattr(model, 'optimizer'):
                mx.eval(model.optimizer.state)

        return batch

    def _apply_update(
        self,
        update_op: Any,  # UpdateOp
        grads: Dict[str, Any],
        loss_val: mx.array,
        state: MutableMapping[str, Any]
    ) -> None:
        """Apply gradients using an UpdateOp's configuration."""
        # Handle gradient scaling for mixed precision
        if update_op.grad_scaler is not None:
            grads = update_op.grad_scaler.unscale(grads)

        # Clip gradients if configured
        if update_op.max_grad_norm is not None:
            grads = self._clip_gradients(grads, update_op.max_grad_norm)

        # Handle gradient accumulation
        if update_op.accumulation_steps > 1:
            update_op._accumulate_gradients(grads)
            if update_op._should_update():
                if update_op.grad_scaler is not None:
                    stepped = update_op.grad_scaler.step(
                        update_op.model.optimizer,
                        update_op.model,
                        update_op._accumulated_grads
                    )
                    if stepped:
                        update_op.grad_scaler.update()
                else:
                    update_op.model.optimizer.update(
                        update_op.model,
                        update_op._accumulated_grads
                    )
                update_op._reset_accumulation()
        else:
            # Standard update
            if update_op.grad_scaler is not None:
                stepped = update_op.grad_scaler.step(
                    update_op.model.optimizer,
                    update_op.model,
                    grads
                )
                if stepped:
                    update_op.grad_scaler.update()
            else:
                update_op.model.optimizer.update(update_op.model, grads)

    def _clip_gradients(self, grads: dict, max_norm: float) -> dict:
        """Clip gradients by global norm."""
        total_norm_sq = mx.array(0.0)

        def accumulate_norm(value):
            nonlocal total_norm_sq
            if isinstance(value, mx.array):
                total_norm_sq = total_norm_sq + mx.sum(value ** 2)
            elif isinstance(value, dict):
                for v in value.values():
                    accumulate_norm(v)

        for v in grads.values():
            accumulate_norm(v)

        total_norm = mx.sqrt(total_norm_sq)
        clip_coef = mx.minimum(max_norm / (total_norm + 1e-6), mx.array(1.0))

        def clip_value(v):
            if isinstance(v, mx.array):
                return v * clip_coef
            elif isinstance(v, dict):
                return {k: clip_value(val) for k, val in v.items()}
            return v

        return {k: clip_value(v) for k, v in grads.items()}

    def get_loss_keys(self) -> Set[str]:
        """Get the set of output keys that are loss values.

        Uses the is_loss property on ops to identify loss operations.
        Falls back to string matching for backwards compatibility.

        Returns:
            Set of key names from loss operations.
        """
        keys: Set[str] = set()
        common_loss_names = {"ce", "loss", "mse", "mae", "bce", "focal", "dice", "hinge"}

        for op in self.ops:
            if isinstance(op, Op) and op.outputs:
                # Use is_loss property if available
                if hasattr(op, "is_loss") and op.is_loss:
                    for k in op.outputs:
                        keys.add(k)
                else:
                    # Fallback to string matching for backwards compatibility
                    for k in op.outputs:
                        k_lower = k.lower()
                        if "loss" in k_lower or k_lower in common_loss_names:
                            keys.add(k)
        return keys

    def validate_ops(self) -> List[str]:
        """Validate the op sequence for potential issues.

        Returns:
            List of warning messages about potential issues.
        """
        warnings = []
        available_keys: Set[str] = set()

        for i, op in enumerate(self.ops):
            if op.outputs:
                for key in op.outputs:
                    if key in available_keys:
                        warnings.append(
                            f"Op {i} ({op.__class__.__name__}) overwrites existing key '{key}'"
                        )
                    available_keys.add(key)

        return warnings
