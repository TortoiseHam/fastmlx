"""Public interface for the :mod:`fastmlx` package."""

from .pipeline import Pipeline
from .network import Network
from .estimator import Estimator
from .util import build

__all__ = ["Pipeline", "Network", "Estimator", "build", "__version__"]

# Package semantic version
__version__: str = "0.1.0"
