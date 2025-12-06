"""Utility helpers for building models."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def build(
    model_fn: Callable[[], nn.Module],
    optimizer_fn: Union[str, Callable[[], optim.Optimizer]] = "adam",
    learning_rate: Optional[float] = None,
    **optimizer_kwargs: Any
) -> nn.Module:
    """Instantiate a model and attach an optimizer.

    Args:
        model_fn: A callable that returns an nn.Module instance.
        optimizer_fn: Either a string name of an optimizer ("adam", "sgd", "adamw",
            "rmsprop", "adagrad") or a callable that returns an optimizer instance.
        learning_rate: Learning rate for the optimizer. If None, uses defaults:
            - adam/adamw: 1e-3
            - sgd: 1e-2
            - rmsprop: 1e-2
            - adagrad: 1e-2
        **optimizer_kwargs: Additional keyword arguments passed to the optimizer.
            Common options include:
            - weight_decay: L2 regularization coefficient
            - betas: (beta1, beta2) for Adam/AdamW
            - momentum: Momentum for SGD
            - eps: Epsilon for numerical stability

    Returns:
        The instantiated model with optimizer attached as model.optimizer.

    Example:
        >>> # Basic usage with default learning rate
        >>> model = build(model_fn=lambda: LeNet(), optimizer_fn="adam")

        >>> # Custom learning rate
        >>> model = build(model_fn=lambda: LeNet(), optimizer_fn="adam", learning_rate=1e-4)

        >>> # SGD with momentum
        >>> model = build(model_fn=lambda: LeNet(), optimizer_fn="sgd",
        ...               learning_rate=0.01, momentum=0.9)

        >>> # AdamW with weight decay
        >>> model = build(model_fn=lambda: LeNet(), optimizer_fn="adamw",
        ...               learning_rate=1e-3, weight_decay=0.01)

        >>> # Custom optimizer function
        >>> model = build(model_fn=lambda: LeNet(),
        ...               optimizer_fn=lambda: optim.Adam(learning_rate=1e-4, betas=(0.9, 0.99)))
    """
    model = model_fn()

    if isinstance(optimizer_fn, str):
        opt_name = optimizer_fn.lower()

        # Set default learning rates
        default_lrs = {
            "adam": 1e-3,
            "adamw": 1e-3,
            "sgd": 1e-2,
            "rmsprop": 1e-2,
            "adagrad": 1e-2,
        }
        lr = learning_rate if learning_rate is not None else default_lrs.get(opt_name, 1e-3)

        # Create optimizer with specified parameters
        if opt_name == "adam":
            optimizer = optim.Adam(learning_rate=lr, **optimizer_kwargs)
        elif opt_name == "adamw":
            optimizer = optim.AdamW(learning_rate=lr, **optimizer_kwargs)
        elif opt_name == "sgd":
            optimizer = optim.SGD(learning_rate=lr, **optimizer_kwargs)
        elif opt_name == "rmsprop":
            optimizer = optim.RMSprop(learning_rate=lr, **optimizer_kwargs)
        elif opt_name == "adagrad":
            optimizer = optim.Adagrad(learning_rate=lr, **optimizer_kwargs)
        else:
            raise ValueError(
                f"Unknown optimizer '{optimizer_fn}'. "
                f"Supported: adam, adamw, sgd, rmsprop, adagrad. "
                f"Or pass a callable that returns an optimizer."
            )
    else:
        # User provided a callable
        optimizer = optimizer_fn()

    # Initialize parameters
    mx.eval(model.parameters())
    model.optimizer = optimizer
    return model
