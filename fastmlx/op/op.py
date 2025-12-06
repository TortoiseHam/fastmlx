"""Base class for pipeline and network operations."""

from __future__ import annotations

from typing import Any, List, MutableMapping, Optional, Sequence, Union


class Op:
    """Base class for all operations in the pipeline and network.

    Args:
        inputs: Names of input keys to read from the data dictionary.
        outputs: Names of output keys to write to the data dictionary.
        mode: When to execute this op. Can be "train", "eval", "test", "infer",
            a list of modes, or None for all modes.
    """

    def __init__(
        self,
        inputs: Optional[Sequence[str]] = None,
        outputs: Optional[Sequence[str]] = None,
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self.inputs: List[str] = [] if inputs is None else ([inputs] if isinstance(inputs, str) else list(inputs))
        self.outputs: List[str] = [] if outputs is None else ([outputs] if isinstance(outputs, str) else list(outputs))
        self.mode: Optional[List[str]] = None if mode is None else ([mode] if isinstance(mode, str) else list(mode))

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Any:
        """Execute the operation.

        Args:
            data: Input data (single array or list of arrays).
            state: Mutable state dictionary for sharing information between ops.

        Returns:
            Output data (single array or tuple of arrays).
        """
        raise NotImplementedError
