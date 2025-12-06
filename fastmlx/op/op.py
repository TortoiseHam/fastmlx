"""Base class for pipeline and network operations."""

from __future__ import annotations

from typing import Any, List, MutableMapping, Optional, Sequence, Set, Union


# All valid execution modes
ALL_MODES = {"train", "eval", "test", "infer"}


def parse_modes(mode: Optional[Union[str, List[str]]]) -> Optional[Set[str]]:
    """Parse mode specification into a set of allowed modes.

    Supports negation syntax: "!train" means all modes except train.

    Args:
        mode: Mode specification. Can be:
            - None: Run in all modes
            - str: Single mode ("train") or negated ("!train")
            - List[str]: Multiple modes, may include negations

    Returns:
        Set of allowed modes, or None if all modes are allowed.

    Example:
        >>> parse_modes("train")
        {'train'}
        >>> parse_modes("!train")
        {'eval', 'test', 'infer'}
        >>> parse_modes(["train", "eval"])
        {'train', 'eval'}
        >>> parse_modes(["!train", "!test"])  # Intersection: exclude train and test
        {'eval', 'infer'}
    """
    if mode is None:
        return None

    modes_list = [mode] if isinstance(mode, str) else list(mode)

    # Separate positive and negative modes
    positive_modes: Set[str] = set()
    negative_modes: Set[str] = set()

    for m in modes_list:
        m = m.strip()
        if m.startswith("!"):
            negative_modes.add(m[1:])
        else:
            positive_modes.add(m)

    # Validate mode names
    all_specified = positive_modes | negative_modes
    invalid = all_specified - ALL_MODES
    if invalid:
        raise ValueError(
            f"Invalid mode(s): {invalid}. Valid modes are: {ALL_MODES}"
        )

    # If only negative modes, start with all and subtract
    if negative_modes and not positive_modes:
        return ALL_MODES - negative_modes

    # If only positive modes, use them
    if positive_modes and not negative_modes:
        return positive_modes

    # If both, use positive modes and subtract negative
    return positive_modes - negative_modes


class Op:
    """Base class for all operations in the pipeline and network.

    Args:
        inputs: Names of input keys to read from the data dictionary.
        outputs: Names of output keys to write to the data dictionary.
        mode: When to execute this op. Can be:
            - None: Run in all modes (default)
            - str: Single mode ("train", "eval", "test", "infer")
            - str with negation: "!train" runs in all modes except train
            - List[str]: Multiple modes, may include negations

    Example:
        >>> # Run only during training
        >>> op = SomeOp(inputs="x", outputs="y", mode="train")

        >>> # Run in train and eval
        >>> op = SomeOp(inputs="x", outputs="y", mode=["train", "eval"])

        >>> # Run in all modes except inference
        >>> op = SomeOp(inputs="x", outputs="y", mode="!infer")

        >>> # Run in all modes except train and test
        >>> op = SomeOp(inputs="x", outputs="y", mode=["!train", "!test"])
    """

    def __init__(
        self,
        inputs: Optional[Sequence[str]] = None,
        outputs: Optional[Sequence[str]] = None,
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        # Normalize inputs/outputs to lists
        if inputs is None:
            self.inputs: List[str] = []
        elif isinstance(inputs, str):
            self.inputs = [inputs]
        else:
            self.inputs = list(inputs)

        if outputs is None:
            self.outputs: List[str] = []
        elif isinstance(outputs, str):
            self.outputs = [outputs]
        else:
            self.outputs = list(outputs)

        # Parse and store mode as a set for efficient lookup
        self._mode_set: Optional[Set[str]] = parse_modes(mode)

        # Also store original for inspection
        self.mode: Optional[List[str]] = (
            None if mode is None else ([mode] if isinstance(mode, str) else list(mode))
        )

    def should_run(self, current_mode: Optional[str]) -> bool:
        """Check if this op should run in the given mode.

        Args:
            current_mode: The current execution mode ("train", "eval", etc.).

        Returns:
            True if the op should run, False otherwise.
        """
        # If no mode restriction, always run
        if self._mode_set is None:
            return True
        # If no current mode specified, run all ops
        if current_mode is None:
            return True
        # Check if current mode is in the op's allowed modes
        return current_mode in self._mode_set

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> Any:
        """Execute the operation.

        Args:
            data: Input data (single array or list of arrays).
            state: Mutable state dictionary for sharing information between ops.

        Returns:
            Output data (single array or tuple of arrays).
        """
        raise NotImplementedError
