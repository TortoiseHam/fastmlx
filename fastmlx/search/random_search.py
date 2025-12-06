"""Random Search for hyperparameter optimization."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .search import Search


class RandomSearch(Search):
    """Random search over hyperparameter space.

    Samples random combinations of hyperparameters for a specified number of trials.
    Supports various distributions for sampling.

    Args:
        search_space: Dictionary defining the search space. Values can be:
            - List: Sample uniformly from the list
            - Tuple (min, max): Sample uniformly from continuous range
            - Dict with 'distribution': Sample from specified distribution
              Supported distributions:
              - {"distribution": "uniform", "low": 0, "high": 1}
              - {"distribution": "log_uniform", "low": 1e-5, "high": 1e-1}
              - {"distribution": "int_uniform", "low": 1, "high": 100}
              - {"distribution": "choice", "options": [a, b, c]}
        num_trials: Number of random samples to evaluate.
        maximize: If True, maximize the objective. If False, minimize.
        seed: Random seed for reproducibility.
        name: Optional name for this search.

    Example:
        >>> search_space = {
        ...     "learning_rate": {"distribution": "log_uniform", "low": 1e-5, "high": 1e-1},
        ...     "batch_size": {"distribution": "choice", "options": [32, 64, 128]},
        ...     "dropout": {"distribution": "uniform", "low": 0.0, "high": 0.5},
        ...     "hidden_dim": {"distribution": "int_uniform", "low": 32, "high": 512}
        ... }
        >>> search = RandomSearch(search_space, num_trials=50, maximize=True)
        >>> results = search.run(train_and_evaluate)
    """

    def __init__(
        self,
        search_space: Dict[str, Any],
        num_trials: int = 20,
        maximize: bool = True,
        seed: Optional[int] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(search_space, maximize, name or "RandomSearch")
        self.num_trials = num_trials
        self._trials_done = 0

        if seed is not None:
            random.seed(seed)

        # Build samplers for each parameter
        self._samplers: Dict[str, Callable[[], Any]] = {}
        for key, spec in search_space.items():
            self._samplers[key] = self._build_sampler(key, spec)

    def _build_sampler(self, key: str, spec: Any) -> Callable[[], Any]:
        """Build a sampler function for a parameter specification."""
        if isinstance(spec, list):
            # Uniform choice from list
            return lambda s=spec: random.choice(s)

        elif isinstance(spec, tuple) and len(spec) == 2:
            # Uniform continuous range
            low, high = spec
            return lambda l=low, h=high: random.uniform(l, h)

        elif isinstance(spec, dict):
            dist = spec.get("distribution", "uniform")

            if dist == "uniform":
                low = spec.get("low", 0.0)
                high = spec.get("high", 1.0)
                return lambda l=low, h=high: random.uniform(l, h)

            elif dist == "log_uniform":
                import math
                low = spec.get("low", 1e-5)
                high = spec.get("high", 1e-1)
                log_low = math.log(low)
                log_high = math.log(high)
                return lambda ll=log_low, lh=log_high: math.exp(random.uniform(ll, lh))

            elif dist == "int_uniform":
                low = spec.get("low", 0)
                high = spec.get("high", 100)
                return lambda l=low, h=high: random.randint(l, h)

            elif dist == "choice":
                options = spec.get("options", [])
                return lambda o=options: random.choice(o)

            elif dist == "normal":
                mean = spec.get("mean", 0.0)
                std = spec.get("std", 1.0)
                return lambda m=mean, s=std: random.gauss(m, s)

            else:
                raise ValueError(f"Unknown distribution: {dist}")

        else:
            raise ValueError(f"Invalid search space specification for '{key}': {spec}")

    def _get_next_params(self) -> Optional[Dict[str, Any]]:
        """Sample a random set of hyperparameters."""
        if self._trials_done >= self.num_trials:
            return None

        self._trials_done += 1
        return {key: sampler() for key, sampler in self._samplers.items()}

    @property
    def total_trials(self) -> int:
        """Return the total number of trials to run."""
        return self.num_trials
