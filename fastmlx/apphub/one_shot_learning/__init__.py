"""One-shot and metric learning examples.

- siamese_mnist: Siamese network for one-shot learning
- triplet_loss: Triplet loss for embedding learning
"""

from . import (
    siamese_mnist,
    triplet_loss,
)

__all__ = [
    "siamese_mnist",
    "triplet_loss",
]
