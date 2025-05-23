import mlx.nn as nn
import mlx.optimizers as optim
import mlx.core as mx


def build(model_fn, optimizer_fn="adam", **kwargs):
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
