"""Random seed utilities for MLX."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

# Store the current seed for retrieval
_current_seed: Optional[int] = None


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    This sets the seed for MLX's random number generator.

    Args:
        seed: Integer seed value.

    Example:
        >>> set_seed(42)
        >>> x = mx.random.normal((3,))
        >>> set_seed(42)
        >>> y = mx.random.normal((3,))
        >>> # x and y will be identical
    """
    global _current_seed
    _current_seed = seed
    mx.random.seed(seed)


def get_seed() -> Optional[int]:
    """Get the current random seed.

    Returns:
        The last set seed, or None if no seed has been set.

    Note:
        This returns the seed that was last set via set_seed(),
        not necessarily the current internal state of the RNG.
    """
    return _current_seed


def manual_seed(seed: int) -> None:
    """Alias for set_seed for compatibility.

    Args:
        seed: Integer seed value.
    """
    set_seed(seed)


class fork_rng:
    """Context manager to fork the random state.

    Creates a new random state within the context that doesn't
    affect the global state.

    Example:
        >>> set_seed(42)
        >>> x1 = mx.random.normal((3,))
        >>> with fork_rng(seed=123):
        ...     y = mx.random.normal((3,))  # Uses seed 123
        >>> x2 = mx.random.normal((3,))  # Continues from x1's state
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize fork context.

        Args:
            seed: Optional seed for the forked state.
                  If None, uses a random seed.
        """
        self.seed = seed
        self._saved_seed: Optional[int] = None

    def __enter__(self):
        """Enter the context and fork RNG state."""
        global _current_seed
        self._saved_seed = _current_seed

        if self.seed is not None:
            mx.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        """Exit the context and restore RNG state."""
        global _current_seed
        if self._saved_seed is not None:
            mx.random.seed(self._saved_seed)
            _current_seed = self._saved_seed


def random_split(key: Optional[int] = None, num: int = 2) -> list:
    """Split a random key into multiple independent keys.

    This is useful for parallel random number generation.

    Args:
        key: Base seed to split. If None, generates random seeds.
        num: Number of keys to generate.

    Returns:
        List of seed values.

    Example:
        >>> keys = random_split(42, num=4)
        >>> # Use each key for independent random operations
    """
    if key is not None:
        mx.random.seed(key)

    # Generate independent seeds
    seeds = []
    for _ in range(num):
        # Generate a random integer to use as a seed
        val = mx.random.randint(0, 2**31 - 1, shape=(1,))
        mx.eval(val)
        seeds.append(int(val.item()))

    return seeds
