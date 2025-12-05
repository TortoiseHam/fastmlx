"""Tracing utilities for FastMLX."""

from .base import Trace

# Metric traces
from .metric import (
    Accuracy,
    LossMonitor,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    MCC,
    Dice,
)

# IO traces
from .io import (
    BestModelSaver,
    ModelSaver,
    CSVLogger,
    ProgressLogger,
    Timer,
)

# Adapt traces
from .adapt import (
    LRScheduler,
    EarlyStopping,
    ReduceLROnPlateau,
    TerminateOnNaN,
    WarmupScheduler,
)

# TensorBoard traces (optional dependency)
try:
    from .tensorboard import TensorBoardLogger, TensorBoardEmbedding
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

__all__ = [
    # Base
    "Trace",
    # Metrics
    "Accuracy",
    "LossMonitor",
    "Precision",
    "Recall",
    "F1Score",
    "ConfusionMatrix",
    "MCC",
    "Dice",
    # IO
    "BestModelSaver",
    "ModelSaver",
    "CSVLogger",
    "ProgressLogger",
    "Timer",
    # Adapt
    "LRScheduler",
    "EarlyStopping",
    "ReduceLROnPlateau",
    "TerminateOnNaN",
    "WarmupScheduler",
]

# Add TensorBoard traces if available
if _TENSORBOARD_AVAILABLE:
    __all__.extend(["TensorBoardLogger", "TensorBoardEmbedding"])
