"""Learning rate schedules and scheduling utilities for FastMLX."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, TypeVar, Union

T = TypeVar('T')


# ====================
# Learning Rate Schedules
# ====================

def cosine_decay(step: int, cycle_length: int, init_lr: float, min_lr: float = 0.0) -> float:
    """Compute a cosine decayed learning rate.

    Args:
        step: Current training step.
        cycle_length: Number of steps in one cosine cycle.
        init_lr: Initial (maximum) learning rate.
        min_lr: Minimum learning rate. Defaults to 0.

    Returns:
        The learning rate for the current step.
    """
    t = (step % cycle_length) / cycle_length
    return min_lr + (init_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t))


def linear_decay(step: int, total_steps: int, init_lr: float, end_lr: float = 0.0) -> float:
    """Compute a linearly decayed learning rate.

    Args:
        step: Current training step.
        total_steps: Total number of training steps.
        init_lr: Initial learning rate.
        end_lr: Final learning rate. Defaults to 0.

    Returns:
        The learning rate for the current step.
    """
    if step >= total_steps:
        return end_lr
    return init_lr - (init_lr - end_lr) * (step / total_steps)


def step_decay(
    step: int,
    init_lr: float,
    decay_rate: float = 0.1,
    decay_steps: int = 1000
) -> float:
    """Compute a step-decayed learning rate.

    Learning rate is reduced by decay_rate every decay_steps steps.

    Args:
        step: Current training step.
        init_lr: Initial learning rate.
        decay_rate: Factor by which to reduce the learning rate.
        decay_steps: Number of steps between each decay.

    Returns:
        The learning rate for the current step.
    """
    return init_lr * (decay_rate ** (step // decay_steps))


def exponential_decay(
    step: int,
    init_lr: float,
    decay_rate: float = 0.96,
    decay_steps: int = 1000
) -> float:
    """Compute an exponentially decayed learning rate.

    Args:
        step: Current training step.
        init_lr: Initial learning rate.
        decay_rate: Decay rate.
        decay_steps: Number of steps for one decay cycle.

    Returns:
        The learning rate for the current step.
    """
    return init_lr * (decay_rate ** (step / decay_steps))


def polynomial_decay(
    step: int,
    total_steps: int,
    init_lr: float,
    end_lr: float = 0.0001,
    power: float = 1.0
) -> float:
    """Compute a polynomial decayed learning rate.

    Args:
        step: Current training step.
        total_steps: Total number of training steps.
        init_lr: Initial learning rate.
        end_lr: Final learning rate.
        power: The power of the polynomial. Defaults to 1 (linear).

    Returns:
        The learning rate for the current step.
    """
    if step >= total_steps:
        return end_lr
    decay = (1 - step / total_steps) ** power
    return (init_lr - end_lr) * decay + end_lr


def warmup_cosine_decay(
    step: int,
    warmup_steps: int,
    total_steps: int,
    init_lr: float,
    min_lr: float = 0.0
) -> float:
    """Compute learning rate with linear warmup followed by cosine decay.

    Args:
        step: Current training step.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        init_lr: Peak learning rate (reached at end of warmup).
        min_lr: Minimum learning rate after decay.

    Returns:
        The learning rate for the current step.
    """
    if step < warmup_steps:
        # Linear warmup
        return init_lr * (step / warmup_steps)
    else:
        # Cosine decay
        decay_steps = total_steps - warmup_steps
        decay_step = step - warmup_steps
        t = decay_step / decay_steps
        return min_lr + (init_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t))


def one_cycle(
    step: int,
    total_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos"
) -> float:
    """Implement the 1cycle learning rate policy.

    The learning rate increases from min_lr to max_lr in the first pct_start
    fraction of steps, then decreases back to min_lr.

    Args:
        step: Current training step.
        total_steps: Total number of training steps.
        max_lr: Maximum learning rate.
        min_lr: Minimum learning rate.
        pct_start: Fraction of total steps for the increasing phase.
        anneal_strategy: 'cos' for cosine annealing, 'linear' for linear.

    Returns:
        The learning rate for the current step.

    Reference:
        Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks
        Using Large Learning Rates", 2019.
    """
    up_steps = int(total_steps * pct_start)
    down_steps = total_steps - up_steps

    if step < up_steps:
        # Increasing phase
        pct = step / up_steps
        if anneal_strategy == "cos":
            return min_lr + (max_lr - min_lr) * 0.5 * (1 - math.cos(math.pi * pct))
        else:
            return min_lr + (max_lr - min_lr) * pct
    else:
        # Decreasing phase
        pct = (step - up_steps) / down_steps
        if anneal_strategy == "cos":
            return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * pct))
        else:
            return max_lr - (max_lr - min_lr) * pct


# ====================
# Scheduler Classes
# ====================

class Scheduler:
    """Base class for schedulers that vary values over training.

    This can be used for learning rate scheduling, data augmentation
    parameters, or any other value that changes during training.
    """

    def get_current_value(self, epoch: int) -> Any:
        """Get the value for the current epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            The scheduled value for this epoch.
        """
        raise NotImplementedError


class EpochScheduler(Scheduler):
    """Schedule values to change at specific epochs.

    Args:
        epoch_dict: Dictionary mapping epoch numbers to values.
                   The value at epoch N is used until the next specified epoch.

    Example:
        >>> scheduler = EpochScheduler({1: 0.1, 10: 0.01, 20: 0.001})
        >>> scheduler.get_current_value(5)   # Returns 0.1
        >>> scheduler.get_current_value(15)  # Returns 0.01
    """

    def __init__(self, epoch_dict: Dict[int, T]) -> None:
        if not epoch_dict:
            raise ValueError("epoch_dict cannot be empty")
        self.epochs = sorted(epoch_dict.keys())
        self.values = {k: epoch_dict[k] for k in self.epochs}

    def get_current_value(self, epoch: int) -> T:
        """Get the value for the current epoch."""
        current_value = self.values[self.epochs[0]]
        for e in self.epochs:
            if epoch >= e:
                current_value = self.values[e]
            else:
                break
        return current_value


class RepeatScheduler(Scheduler):
    """Repeat a sequence of values cyclically.

    Args:
        values: List of values to cycle through.
        repeat_per_epoch: If True, advance value each epoch.
                         If False, advance each step.

    Example:
        >>> scheduler = RepeatScheduler([0.1, 0.01, 0.001])
        >>> scheduler.get_current_value(0)  # Returns 0.1
        >>> scheduler.get_current_value(1)  # Returns 0.01
        >>> scheduler.get_current_value(3)  # Returns 0.1 (cycles back)
    """

    def __init__(self, values: List[T], repeat_per_epoch: bool = True) -> None:
        if not values:
            raise ValueError("values cannot be empty")
        self.values = values
        self.repeat_per_epoch = repeat_per_epoch

    def get_current_value(self, epoch: int) -> T:
        """Get the value for the current epoch."""
        idx = epoch % len(self.values)
        return self.values[idx]


class LambdaScheduler(Scheduler):
    """Use a custom function to compute scheduled values.

    Args:
        fn: A function that takes an epoch number and returns a value.

    Example:
        >>> scheduler = LambdaScheduler(lambda epoch: 0.1 * (0.9 ** epoch))
        >>> scheduler.get_current_value(10)  # Returns ~0.035
    """

    def __init__(self, fn: Callable[[int], T]) -> None:
        self.fn = fn

    def get_current_value(self, epoch: int) -> T:
        """Get the value for the current epoch."""
        return self.fn(epoch)


# ====================
# Utility Functions
# ====================

def get_current_items(
    items: Union[T, Scheduler, List[Union[T, Scheduler]]],
    epoch: int
) -> Union[T, List[T]]:
    """Get current values from items that may be scheduled.

    Args:
        items: A value, Scheduler, or list of values/Schedulers.
        epoch: Current epoch number.

    Returns:
        The current value(s) for this epoch.
    """
    if isinstance(items, Scheduler):
        return items.get_current_value(epoch)
    elif isinstance(items, list):
        return [
            item.get_current_value(epoch) if isinstance(item, Scheduler) else item
            for item in items
        ]
    else:
        return items


def get_signature_epochs(items: Union[Any, Scheduler, List[Any]]) -> List[int]:
    """Get all epochs where scheduled values change.

    Args:
        items: A value, Scheduler, or list of values/Schedulers.

    Returns:
        Sorted list of epochs where values change.
    """
    epochs = set()

    def collect_epochs(item):
        if isinstance(item, EpochScheduler):
            epochs.update(item.epochs)
        elif isinstance(item, list):
            for i in item:
                collect_epochs(i)

    collect_epochs(items)
    return sorted(epochs)
