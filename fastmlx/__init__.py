"""FastMLX: A deep learning framework based on MLX with an API inspired by FastEstimator.

FastMLX provides a high-level API for building and training neural networks on Apple Silicon,
combining the performance of MLX with the intuitive design patterns of FastEstimator.

Example:
    >>> import fastmlx as fm
    >>> from fastmlx.op import Minmax, CrossEntropy, ModelOp, UpdateOp
    >>> from fastmlx.trace import Accuracy, BestModelSaver
    >>> from fastmlx.architecture import LeNet
    >>>
    >>> # Create pipeline
    >>> pipeline = fm.Pipeline(train_data=train_ds, eval_data=eval_ds, batch_size=32,
    ...                        ops=[Minmax(inputs="x", outputs="x")])
    >>>
    >>> # Build model
    >>> model = fm.build(LeNet, optimizer_fn="adam")
    >>>
    >>> # Create network
    >>> network = fm.Network([
    ...     ModelOp(model=model, inputs="x", outputs="y_pred"),
    ...     CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    ...     UpdateOp(model=model, loss_name="ce")
    ... ])
    >>>
    >>> # Train
    >>> estimator = fm.Estimator(pipeline=pipeline, network=network, epochs=10,
    ...                          traces=[Accuracy(true_key="y", pred_key="y_pred")])
    >>> estimator.fit()
"""

# Import submodules for easy access
from . import architecture, backend, dataset, op, schedule, search, summary, trace
from .backend.amp import AMPConfig
from .display import BatchDisplay, GridDisplay
from .estimator import Estimator
from .network import Network
from .pipeline import Pipeline
from .util import build

__all__ = [
    # Core classes
    "Pipeline",
    "Network",
    "Estimator",
    "build",
    # AMP support
    "AMPConfig",
    # Display utilities
    "BatchDisplay",
    "GridDisplay",
    # Submodules
    "op",
    "trace",
    "architecture",
    "dataset",
    "schedule",
    "search",
    "summary",
    "backend",
    # Version
    "__version__",
]

# Package semantic version
__version__: str = "0.2.0"
