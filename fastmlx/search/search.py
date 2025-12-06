"""Base Search class for hyperparameter optimization."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class SearchResults:
    """Container for hyperparameter search results.

    Attributes:
        search_id: Unique identifier for this search run.
        best_params: The best hyperparameters found.
        best_score: The best score achieved.
        all_results: List of (params, score) tuples for all trials.
        search_space: The search space definition.
        maximize: Whether the search was maximizing or minimizing.
        duration: Total search duration in seconds.
    """

    search_id: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)
    search_space: Dict[str, Any] = field(default_factory=dict)
    maximize: bool = True
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "search_id": self.search_id,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "all_results": self.all_results,
            "search_space": self.search_space,
            "maximize": self.maximize,
            "duration": self.duration,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "SearchResults":
        """Load results from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            search_id=data["search_id"],
            best_params=data["best_params"],
            best_score=data["best_score"],
            all_results=[tuple(r) for r in data["all_results"]],
            search_space=data.get("search_space", {}),
            maximize=data.get("maximize", True),
            duration=data.get("duration", 0.0),
        )

    def summary(self) -> str:
        """Return a human-readable summary of results."""
        lines = [
            f"Search ID: {self.search_id}",
            f"Best Score: {self.best_score:.6f} ({'max' if self.maximize else 'min'})",
            "Best Parameters:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"Total Trials: {len(self.all_results)}")
        lines.append(f"Duration: {self.duration:.1f}s")
        return "\n".join(lines)


class Search(ABC):
    """Abstract base class for hyperparameter search algorithms.

    Subclasses must implement the `_get_next_params` method to generate
    the next set of hyperparameters to evaluate.

    Args:
        search_space: Dictionary defining the hyperparameter search space.
                     Keys are parameter names, values define the search range.
        maximize: If True, maximize the objective. If False, minimize.
        name: Optional name for this search.

    Example:
        >>> class MySearch(Search):
        ...     def _get_next_params(self) -> Optional[Dict[str, Any]]:
        ...         # Return next params to try, or None if search is complete
        ...         pass
        ...
        >>> search = MySearch({"lr": [0.001, 0.01, 0.1]})
        >>> results = search.run(train_fn)
    """

    def __init__(
        self,
        search_space: Dict[str, Any],
        maximize: bool = True,
        name: Optional[str] = None
    ) -> None:
        self.search_space = search_space
        self.maximize = maximize
        self.name = name or self.__class__.__name__
        self.search_id = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._results: List[Tuple[Dict[str, Any], float]] = []
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_score: Optional[float] = None
        self._current_trial: int = 0

    @abstractmethod
    def _get_next_params(self) -> Optional[Dict[str, Any]]:
        """Generate the next set of hyperparameters to evaluate.

        Returns:
            Dictionary of hyperparameters, or None if search is complete.
        """
        raise NotImplementedError

    def _update_best(self, params: Dict[str, Any], score: float) -> None:
        """Update best parameters if this score is better."""
        if self._best_score is None:
            self._best_params = params
            self._best_score = score
        elif self.maximize and score > self._best_score:
            self._best_params = params
            self._best_score = score
        elif not self.maximize and score < self._best_score:
            self._best_params = params
            self._best_score = score

    def run(
        self,
        train_fn: Callable[[Dict[str, Any]], float],
        save_dir: Optional[str] = None,
        verbose: bool = True
    ) -> SearchResults:
        """Run the hyperparameter search.

        Args:
            train_fn: A function that takes hyperparameters dict and returns a score.
            save_dir: Optional directory to save results.
            verbose: If True, print progress.

        Returns:
            SearchResults containing the best parameters and all trial results.
        """
        start_time = datetime.now()

        while True:
            params = self._get_next_params()
            if params is None:
                break

            self._current_trial += 1

            if verbose:
                print(f"\n[{self.name}] Trial {self._current_trial}: {params}")

            try:
                score = train_fn(params)
            except Exception as e:
                if verbose:
                    print(f"  Trial failed: {e}")
                score = float("-inf") if self.maximize else float("inf")

            self._results.append((params, score))
            self._update_best(params, score)

            if verbose:
                print(f"  Score: {score:.6f}")
                if self._best_score is not None:
                    print(f"  Best so far: {self._best_score:.6f}")

        duration = (datetime.now() - start_time).total_seconds()

        results = SearchResults(
            search_id=self.search_id,
            best_params=self._best_params or {},
            best_score=self._best_score or 0.0,
            all_results=self._results,
            search_space=self.search_space,
            maximize=self.maximize,
            duration=duration,
        )

        if save_dir:
            results.save(os.path.join(save_dir, f"{self.search_id}_results.json"))

        if verbose:
            print(f"\n{'='*50}")
            print(results.summary())

        return results

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Return the best parameters found so far."""
        return self._best_params

    @property
    def best_score(self) -> Optional[float]:
        """Return the best score found so far."""
        return self._best_score
