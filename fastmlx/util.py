"""Utility helpers for building models."""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def build(model_fn: Callable[[], nn.Module], optimizer_fn: str | Callable[[], optim.Optimizer] = "adam", **kwargs) -> nn.Module:
    """Instantiate a model and optimizer."""
    model = model_fn()
    if isinstance(optimizer_fn, str):
        if optimizer_fn.lower() == "adam":
            optimizer = optim.Adam(learning_rate=1e-3)
        elif optimizer_fn.lower() == "sgd":
            optimizer = optim.SGD(learning_rate=1e-2)
        else:
            raise ValueError(f"Unknown optimizer {optimizer_fn}")
    else:
        optimizer = optimizer_fn()
    # initialize parameters
    mx.eval(model.parameters())
    model.optimizer = optimizer
    return model
