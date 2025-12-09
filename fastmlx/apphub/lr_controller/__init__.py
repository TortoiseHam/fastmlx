"""Learning rate controller examples.

- super_convergence: 1cycle learning rate policy for fast training
- lr_finder: Learning rate range test
"""

from . import (
    lr_finder,
    super_convergence,
)

__all__ = [
    "super_convergence",
    "lr_finder",
]
