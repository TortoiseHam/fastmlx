"""Tracing utilities for FastMLX."""

# Adapt traces
from .adapt import (
    EarlyStopping,
    LRScheduler,
    ReduceLROnPlateau,
    TerminateOnNaN,
    WarmupScheduler,
)
from .base import Trace

# IO traces
from .io import (
    BestModelSaver,
    CSVLogger,
    ImageSaver,
    ImageViewer,
    ModelSaver,
    ProgressLogger,
    Timer,
)

# Metric traces
from .metric import (
    MCC,
    AUC,
    Accuracy,
    ConfusionMatrix,
    Dice,
    F1Score,
    LossMonitor,
    Precision,
    Recall,
)

# XAI traces
from .xai import (
    GradCAM,
    IntegratedGradients,
    Saliency,
)

# TensorBoard traces (optional dependency)
try:
    from .tensorboard import TensorBoardEmbedding, TensorBoardLogger  # noqa: F401
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

__all__ = [
    # Base
    "Trace",
    # Metrics
    "Accuracy",
    "AUC",
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
    "ImageSaver",
    "ImageViewer",
    # Adapt
    "LRScheduler",
    "EarlyStopping",
    "ReduceLROnPlateau",
    "TerminateOnNaN",
    "WarmupScheduler",
    # XAI
    "GradCAM",
    "Saliency",
    "IntegratedGradients",
]

# Add TensorBoard traces if available
if _TENSORBOARD_AVAILABLE:
    __all__.extend(["TensorBoardLogger", "TensorBoardEmbedding"])
