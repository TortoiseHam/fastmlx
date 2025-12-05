"""Experiment class for comprehensive experiment tracking."""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .summary import Summary


@dataclass
class Experiment:
    """Comprehensive experiment tracking with configuration and environment info.

    Experiment wraps a Summary with additional context about the training run,
    including hyperparameters, system info, and configuration.

    Args:
        name: Name of the experiment.
        description: Optional description.
        tags: Optional list of tags for organization.
        config: Hyperparameters and configuration.

    Example:
        >>> exp = Experiment(
        ...     name="mnist_v1",
        ...     description="LeNet on MNIST with Adam",
        ...     tags=["mnist", "baseline"],
        ...     config={"lr": 0.001, "batch_size": 32, "epochs": 10}
        ... )
        >>> exp.start()
        >>> # ... training loop ...
        >>> exp.log_metric("accuracy", 0.95, epoch=10)
        >>> exp.end()
        >>> exp.save("experiments/")
    """

    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    summary: Summary = field(default_factory=lambda: Summary("experiment"))
    system_info: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "created"  # created, running, completed, failed

    def __post_init__(self):
        self.summary = Summary(self.name)
        self._collect_system_info()

    def _collect_system_info(self) -> None:
        """Collect system information."""
        self.system_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        }

        # Try to get MLX info
        try:
            import mlx.core as mx
            self.system_info["mlx_version"] = mx.__version__ if hasattr(mx, "__version__") else "unknown"
            self.system_info["mlx_default_device"] = str(mx.default_device())
        except ImportError:
            pass

    def start(self) -> "Experiment":
        """Mark the experiment as started.

        Returns:
            Self for chaining.
        """
        self.start_time = datetime.now()
        self.status = "running"
        return self

    def end(self, status: str = "completed") -> "Experiment":
        """Mark the experiment as ended.

        Args:
            status: Final status ("completed", "failed", etc.).

        Returns:
            Self for chaining.
        """
        self.end_time = datetime.now()
        self.status = status
        return self

    def log_metric(
        self,
        metric: str,
        value: float,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        mode: str = "train",
        **kwargs
    ) -> None:
        """Log a metric value.

        Args:
            metric: Name of the metric.
            value: Metric value.
            epoch: Epoch number.
            step: Step number.
            mode: Training mode.
            **kwargs: Additional metadata.
        """
        self.summary.add(metric, value, epoch=epoch, step=step, mode=mode, **kwargs)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        mode: str = "train"
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names to values.
            epoch: Epoch number.
            step: Step number.
            mode: Training mode.
        """
        for name, value in metrics.items():
            self.log_metric(name, value, epoch=epoch, step=step, mode=mode)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Add or update configuration parameters.

        Args:
            params: Dictionary of parameters to add.
        """
        self.config.update(params)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment.

        Args:
            tag: Tag to add.
        """
        if tag not in self.tags:
            self.tags.append(tag)

    @property
    def duration(self) -> Optional[float]:
        """Return experiment duration in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "config": self.config,
            "summary": self.summary.to_dict(),
            "system_info": self.system_info,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "duration": self.duration,
        }

    def save(self, path: str) -> str:
        """Save experiment to a JSON file.

        Args:
            path: Directory or file path.

        Returns:
            Path to the saved file.
        """
        if os.path.isdir(path) or not path.endswith(".json"):
            os.makedirs(path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(path, f"{self.name}_{timestamp}.json")
        else:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            filepath = path

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return filepath

    @classmethod
    def load(cls, path: str) -> "Experiment":
        """Load experiment from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Loaded Experiment instance.
        """
        with open(path, "r") as f:
            data = json.load(f)

        exp = cls(
            name=data["name"],
            description=data.get("description", ""),
            tags=data.get("tags", []),
            config=data.get("config", {}),
        )

        exp.summary = Summary.load(path) if "summary" in data else Summary(exp.name)
        if isinstance(data.get("summary"), dict):
            exp.summary.history = data["summary"].get("history", {})

        exp.system_info = data.get("system_info", {})
        exp.status = data.get("status", "completed")

        if data.get("start_time"):
            exp.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            exp.end_time = datetime.fromisoformat(data["end_time"])

        return exp

    def __repr__(self) -> str:
        return f"Experiment(name='{self.name}', status='{self.status}', tags={self.tags})"

    def describe(self) -> str:
        """Return a human-readable description."""
        lines = [
            f"Experiment: {self.name}",
            "=" * 50,
            f"Description: {self.description}" if self.description else "",
            f"Status: {self.status}",
            f"Tags: {', '.join(self.tags) if self.tags else 'none'}",
            "",
            "Configuration:",
        ]

        for k, v in self.config.items():
            lines.append(f"  {k}: {v}")

        if self.duration:
            lines.append(f"\nDuration: {self.duration:.1f}s")

        lines.append("\n" + self.summary.describe())

        return "\n".join(lines)
