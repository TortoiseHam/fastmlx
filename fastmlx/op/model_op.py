from __future__ import annotations

from typing import Any, List, MutableMapping, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .op import Op


class ModelOp(Op):
    """Forward pass of an :class:`mlx.nn.Module`.

    This op runs the model's forward pass on the input data. During training,
    it sets the model to training mode; during evaluation, it sets eval mode.

    Args:
        model: The MLX module to run.
        inputs: Key(s) to read input data from the batch dictionary.
            If multiple keys are provided, they are passed as separate
            positional arguments to the model.
        outputs: Key(s) to write the model outputs to.
        mode: When to run this op. Default None runs in all modes.

    Example:
        Single input:
        >>> ModelOp(model=model, inputs="x", outputs="y_pred")

        Multiple inputs (e.g., for encoder-decoder):
        >>> ModelOp(model=model, inputs=["encoder_input", "decoder_input"],
        ...         outputs="y_pred")
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[str, List[str]],
        outputs: Union[str, List[str]],
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs, outputs, mode)
        self.model: nn.Module = model

    def forward(self, data: Any, state: MutableMapping[str, Any]) -> mx.array:
        """Run model forward pass.

        Args:
            data: Input data. If multiple inputs were specified, this is a list.
            state: Training state dictionary.

        Returns:
            Model output(s).
        """
        # Set training mode based on current mode
        is_training = state.get("mode") == "train"
        self.model.train(is_training)

        # Handle multiple inputs - unpack as positional arguments
        if isinstance(data, (list, tuple)) and len(self.inputs) > 1:
            return self.model(*data)
        return self.model(data)
