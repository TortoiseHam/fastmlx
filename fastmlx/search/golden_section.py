"""Golden Section Search for 1D hyperparameter optimization."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .search import Search

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2
RESPHI = 2 - PHI  # 1/phi


class GoldenSection(Search):
    """Golden Section search for 1D optimization.

    An efficient algorithm for finding the extremum of a unimodal function
    within a specified interval. Uses the golden ratio to progressively
    narrow the search interval.

    Best suited for optimizing a single continuous hyperparameter.

    Args:
        param_name: Name of the parameter to optimize.
        low: Lower bound of the search interval.
        high: Upper bound of the search interval.
        tolerance: Stop when interval is smaller than this value.
        max_iters: Maximum number of iterations.
        maximize: If True, maximize the objective. If False, minimize.
        name: Optional name for this search.

    Example:
        >>> search = GoldenSection(
        ...     param_name="learning_rate",
        ...     low=1e-5,
        ...     high=1e-1,
        ...     tolerance=1e-6,
        ...     maximize=True
        ... )
        >>> results = search.run(train_and_evaluate)
        >>> print(f"Best LR: {results.best_params['learning_rate']}")

    Note:
        This assumes the objective function is unimodal (has a single peak/valley).
        For multi-modal functions, consider RandomSearch or GridSearch.
    """

    def __init__(
        self,
        param_name: str,
        low: float,
        high: float,
        tolerance: float = 1e-6,
        max_iters: int = 50,
        maximize: bool = True,
        name: Optional[str] = None
    ) -> None:
        search_space = {param_name: (low, high)}
        super().__init__(search_space, maximize, name or "GoldenSection")

        self.param_name = param_name
        self.tolerance = tolerance
        self.max_iters = max_iters

        # Initialize interval bounds
        self._a = low
        self._b = high

        # Interior points
        self._c = self._b - RESPHI * (self._b - self._a)
        self._d = self._a + RESPHI * (self._b - self._a)

        # Scores at interior points (None until evaluated)
        self._fc: Optional[float] = None
        self._fd: Optional[float] = None

        # Current state
        self._iteration = 0
        self._phase = "init_c"  # init_c, init_d, iterate, done

    def _get_next_params(self) -> Optional[Dict[str, Any]]:
        """Get the next parameter value to evaluate."""
        if self._iteration >= self.max_iters:
            self._phase = "done"

        if abs(self._b - self._a) < self.tolerance:
            self._phase = "done"

        if self._phase == "done":
            return None

        self._iteration += 1

        if self._phase == "init_c":
            # First evaluation at point c
            self._phase = "init_d"
            return {self.param_name: self._c}

        elif self._phase == "init_d":
            # Store result from c evaluation
            if self._results:
                self._fc = self._results[-1][1]
            # Evaluate at point d
            self._phase = "iterate"
            return {self.param_name: self._d}

        elif self._phase == "iterate":
            # Store result from previous evaluation
            if len(self._results) >= 2:
                # Last result is for d if fd is None, otherwise for the new point
                if self._fd is None:
                    self._fd = self._results[-1][1]

            # Compare and narrow interval
            if self._fc is None or self._fd is None:
                self._phase = "done"
                return None

            # For maximization, we want to keep the region with higher values
            if self.maximize:
                if self._fc > self._fd:
                    # Maximum is in [a, d]
                    self._b = self._d
                    self._d = self._c
                    self._fd = self._fc
                    self._c = self._b - RESPHI * (self._b - self._a)
                    self._fc = None
                    return {self.param_name: self._c}
                else:
                    # Maximum is in [c, b]
                    self._a = self._c
                    self._c = self._d
                    self._fc = self._fd
                    self._d = self._a + RESPHI * (self._b - self._a)
                    self._fd = None
                    return {self.param_name: self._d}
            else:
                # Minimization
                if self._fc < self._fd:
                    # Minimum is in [a, d]
                    self._b = self._d
                    self._d = self._c
                    self._fd = self._fc
                    self._c = self._b - RESPHI * (self._b - self._a)
                    self._fc = None
                    return {self.param_name: self._c}
                else:
                    # Minimum is in [c, b]
                    self._a = self._c
                    self._c = self._d
                    self._fc = self._fd
                    self._d = self._a + RESPHI * (self._b - self._a)
                    self._fd = None
                    return {self.param_name: self._d}

        return None

    def _update_best(self, params: Dict[str, Any], score: float) -> None:
        """Override to update fc/fd based on which point was evaluated."""
        super()._update_best(params, score)

        # Update interior point scores
        value = params[self.param_name]
        if abs(value - self._c) < 1e-10:
            self._fc = score
        elif abs(value - self._d) < 1e-10:
            self._fd = score

    @property
    def current_interval(self) -> tuple:
        """Return the current search interval."""
        return (self._a, self._b)
