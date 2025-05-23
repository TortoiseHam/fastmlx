"""Base class for pipeline and network operations."""

from __future__ import annotations

from typing import Iterable, List, MutableMapping, Sequence, Any, Optional


class Op:
    def __init__(self, inputs: Optional[Sequence[str]] = None, outputs: Optional[Sequence[str]] = None) -> None:
        self.inputs: List[str] = [] if inputs is None else ([inputs] if isinstance(inputs, str) else list(inputs))
        self.outputs: List[str] = [] if outputs is None else ([outputs] if isinstance(outputs, str) else list(outputs))

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Any:
        raise NotImplementedError
