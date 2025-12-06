"""A simple sequential network abstraction."""

from __future__ import annotations

from typing import Any, List, MutableMapping, Sequence, Set

from .op.op import Op


class NetworkError(Exception):
    """Exception raised for network execution errors."""
    pass


class Network:
    """Execute a sequence of :class:`~fastmlx.op.Op` objects.

    Args:
        ops: Sequence of Op objects to execute in order.

    Example:
        >>> network = Network([
        ...     ModelOp(model=model, inputs="x", outputs="y_pred"),
        ...     CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        ...     UpdateOp(model=model, loss_name="ce")
        ... ])
        >>> result = network.run(batch, state)
    """

    def __init__(self, ops: Sequence[Op]) -> None:
        self.ops: List[Op] = list(ops)

    def _validate_inputs(
        self,
        op: Op,
        store: MutableMapping[str, Any],
        op_index: int
    ) -> None:
        """Validate that all input keys exist in the store.

        Args:
            op: The operation to validate inputs for.
            store: The current data store (batch).
            op_index: Index of the op for error messages.

        Raises:
            NetworkError: If a required input key is missing.
        """
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
        """Execute all ops in sequence on the batch.

        Args:
            batch: Dictionary containing input data.
            state: Dictionary for sharing state between ops and traces.
                   Should contain 'mode' key for mode filtering.

        Returns:
            The modified batch dictionary with outputs added.

        Raises:
            NetworkError: If an op fails or required inputs are missing.
        """
        store = batch
        current_mode = state.get("mode")

        for i, op in enumerate(self.ops):
            # Check if op should run in current mode
            if hasattr(op, "should_run") and not op.should_run(current_mode):
                continue

            # Validate inputs exist
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

            # Execute forward pass
            try:
                out = op.forward(inp, state)
            except Exception as e:
                raise NetworkError(
                    f"Op {i} ({op.__class__.__name__}): Forward pass failed with error: {e}"
                ) from e

            # Store outputs
            if op.outputs:
                if len(op.outputs) == 1:
                    store[op.outputs[0]] = out
                else:
                    if out is None:
                        raise NetworkError(
                            f"Op {i} ({op.__class__.__name__}): Expected {len(op.outputs)} "
                            f"outputs but got None"
                        )
                    try:
                        for k, v in zip(op.outputs, out):
                            store[k] = v
                    except TypeError as e:
                        raise NetworkError(
                            f"Op {i} ({op.__class__.__name__}): Output not iterable. "
                            f"Expected {len(op.outputs)} outputs."
                        ) from e

        return store

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
            # Check if inputs will be available
            for key in op.inputs:
                if key not in available_keys and i > 0:
                    # This input must come from the batch
                    pass

            # Add outputs to available keys
            if op.outputs:
                for key in op.outputs:
                    if key in available_keys:
                        warnings.append(
                            f"Op {i} ({op.__class__.__name__}) overwrites existing key '{key}'"
                        )
                    available_keys.add(key)

        return warnings
