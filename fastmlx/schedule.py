"""Learning rate schedules."""

import math

def cosine_decay(step: int, cycle_length: int, init_lr: float) -> float:
    """Compute a cosine decayed learning rate."""
    t = (step % cycle_length) / cycle_length
    return init_lr * 0.5 * (1 + math.cos(math.pi * t))
