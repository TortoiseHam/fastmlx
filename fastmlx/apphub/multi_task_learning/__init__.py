"""Multi-task learning examples.

- multitask_learning: Uncertainty-weighted multi-task learning
- knowledge_distillation: Teacher-student model training
"""

from . import (
    knowledge_distillation,
    multitask_learning,
)

__all__ = [
    "multitask_learning",
    "knowledge_distillation",
]
