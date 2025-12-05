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
