"""Summary class for tracking training metrics and history."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class Summary:
    """Container for tracking and storing training metrics.

    Summary collects metrics during training and provides utilities for
    accessing, saving, and visualizing the training history.

    Attributes:
        name: Name of this training run.
        history: Dictionary mapping metric names to lists of values.
        metadata: Additional metadata about the run.

    Example:
        >>> summary = Summary("mnist_experiment")
        >>> summary.add("loss", 0.5, epoch=1, mode="train")
        >>> summary.add("accuracy", 0.95, epoch=1, mode="eval")
        >>> summary.save("results/")
    """

    name: str
    history: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.history, defaultdict):
            self.history = defaultdict(list, self.history)

    def add(
        self,
        metric: str,
        value: Union[float, int],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        mode: str = "train",
        **kwargs: Any
    ) -> None:
        """Add a metric value to the history.

        Args:
            metric: Name of the metric (e.g., "loss", "accuracy").
            value: The metric value.
            epoch: Epoch number.
            step: Step/batch number.
            mode: Training mode ("train", "eval", "test").
            **kwargs: Additional metadata for this entry.
        """
        entry = {
            "value": value,
            "epoch": epoch,
            "step": step,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.history[metric].append(entry)

    def get(
        self,
        metric: str,
        mode: Optional[str] = None,
        epoch: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get metric values from history.

        Args:
            metric: Name of the metric.
            mode: Filter by mode ("train", "eval", "test").
            epoch: Filter by epoch.

        Returns:
            List of metric entries matching the filters.
        """
        entries = self.history.get(metric, [])

        if mode is not None:
            entries = [e for e in entries if e.get("mode") == mode]

        if epoch is not None:
            entries = [e for e in entries if e.get("epoch") == epoch]

        return entries

    def get_values(
        self,
        metric: str,
        mode: Optional[str] = None
    ) -> List[float]:
        """Get just the values for a metric.

        Args:
            metric: Name of the metric.
            mode: Filter by mode.

        Returns:
            List of metric values.
        """
        entries = self.get(metric, mode=mode)
        return [e["value"] for e in entries]

    def get_best(
        self,
        metric: str,
        mode: str = "eval",
        maximize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get the best entry for a metric.

        Args:
            metric: Name of the metric.
            mode: Filter by mode.
            maximize: If True, return max. If False, return min.

        Returns:
            The best entry, or None if no entries exist.
        """
        entries = self.get(metric, mode=mode)
        if not entries:
            return None

        if maximize:
            return max(entries, key=lambda e: e["value"])
        else:
            return min(entries, key=lambda e: e["value"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to a dictionary."""
        return {
            "name": self.name,
            "history": dict(self.history),
            "metadata": self.metadata,
        }

    def save(self, path: str) -> str:
        """Save summary to a JSON file.

        Args:
            path: Directory or file path. If directory, creates a file
                  named "{name}_summary.json".

        Returns:
            The path to the saved file.
        """
        if os.path.isdir(path) or not path.endswith(".json"):
            os.makedirs(path, exist_ok=True)
            filepath = os.path.join(path, f"{self.name}_summary.json")
        else:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            filepath = path

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return filepath

    @classmethod
    def load(cls, path: str) -> "Summary":
        """Load a summary from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Loaded Summary instance.
        """
        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            name=data["name"],
            history=defaultdict(list, data.get("history", {})),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        metrics = list(self.history.keys())
        total_entries = sum(len(v) for v in self.history.values())
        return f"Summary(name='{self.name}', metrics={metrics}, entries={total_entries})"

    def describe(self) -> str:
        """Return a human-readable description of the summary."""
        lines = [f"Summary: {self.name}", "=" * 40]

        for metric, entries in self.history.items():
            if entries:
                values = [e["value"] for e in entries]
                train_values = [e["value"] for e in entries if e.get("mode") == "train"]
                eval_values = [e["value"] for e in entries if e.get("mode") == "eval"]

                lines.append(f"\n{metric}:")
                lines.append(f"  Total entries: {len(entries)}")
                if train_values:
                    lines.append(f"  Train - min: {min(train_values):.4f}, max: {max(train_values):.4f}")
                if eval_values:
                    lines.append(f"  Eval  - min: {min(eval_values):.4f}, max: {max(eval_values):.4f}")

        return "\n".join(lines)
