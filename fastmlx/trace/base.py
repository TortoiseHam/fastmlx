"""Base class for all traces (callbacks)."""

from __future__ import annotations

from typing import Any, List, MutableMapping, Optional, Set, Union


class Trace:
    """Base class for traces (callbacks) that hook into the training lifecycle.

    Traces are called at specific points during training to monitor progress,
    adapt training behavior, save checkpoints, and log metrics.

    Args:
        inputs: Keys that this trace reads from batch/state. Used for documentation
            and potential future validation.
        outputs: Keys that this trace writes to state['metrics']. Used for documentation.
        mode: Mode(s) in which this trace should run. Can be:
            - None: Run in all modes (default)
            - str: Single mode ("train", "eval", "test", "infer")
            - List[str]: Multiple modes (["train", "eval"])

    Lifecycle Hooks:
        - on_start(state): Called once at the beginning of training
        - on_epoch_begin(state): Called at the start of each epoch
        - on_batch_begin(batch, state): Called before each batch
        - on_batch_end(batch, state): Called after each batch
        - on_epoch_end(state): Called at the end of each epoch
        - on_finish(state): Called once after all training is complete

    Example:
        >>> class MyTrace(Trace):
        ...     def __init__(self, monitor_key: str):
        ...         super().__init__(inputs=[monitor_key], mode="train")
        ...         self.monitor_key = monitor_key
        ...
        ...     def on_batch_end(self, batch, state):
        ...         value = batch.get(self.monitor_key)
        ...         print(f"{self.monitor_key}: {value}")
    """

    def __init__(
        self,
        inputs: Optional[Union[str, List[str]]] = None,
        outputs: Optional[Union[str, List[str]]] = None,
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

        # Normalize mode to set for fast lookup
        if mode is None:
            self.mode: Optional[Set[str]] = None
        elif isinstance(mode, str):
            self.mode = {mode}
        else:
            self.mode = set(mode)

    def should_run(self, current_mode: Optional[str]) -> bool:
        """Check if this trace should run in the given mode.

        Args:
            current_mode: The current execution mode ("train", "eval", etc.).

        Returns:
            True if the trace should run, False otherwise.
        """
        # If no mode restriction, always run
        if self.mode is None:
            return True
        # If no current mode specified, run all traces
        if current_mode is None:
            return True
        # Check if current mode is in the trace's allowed modes
        return current_mode in self.mode

    def on_start(self, state: MutableMapping[str, Any]) -> None:
        """Called once at the beginning of training.

        Args:
            state: Training state dictionary.
        """
        pass

    def on_epoch_begin(self, state: MutableMapping[str, Any]) -> None:
        """Called at the start of each epoch.

        Args:
            state: Training state dictionary containing 'epoch', 'mode', etc.
        """
        pass

    def on_batch_begin(
        self,
        batch: MutableMapping[str, Any],
        state: MutableMapping[str, Any]
    ) -> None:
        """Called before each batch is processed.

        Args:
            batch: The current batch data dictionary.
            state: Training state dictionary.
        """
        pass

    def on_batch_end(
        self,
        batch: MutableMapping[str, Any],
        state: MutableMapping[str, Any]
    ) -> None:
        """Called after each batch is processed.

        Args:
            batch: The current batch data dictionary (includes model outputs).
            state: Training state dictionary.
        """
        pass

    def on_epoch_end(self, state: MutableMapping[str, Any]) -> None:
        """Called at the end of each epoch.

        Args:
            state: Training state dictionary containing 'metrics', etc.
        """
        pass

    def on_finish(self, state: MutableMapping[str, Any]) -> None:
        """Called once after all training is complete.

        Args:
            state: Final training state dictionary.
        """
        pass
