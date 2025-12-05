"""Grid Search for hyperparameter optimization."""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterator, List, Optional

from .search import Search


class GridSearch(Search):
    """Exhaustive grid search over hyperparameter space.

    Evaluates all combinations of the specified hyperparameter values.

    Args:
        search_space: Dictionary where keys are parameter names and values are
                     lists of values to try for each parameter.
        maximize: If True, maximize the objective. If False, minimize.
        name: Optional name for this search.

    Example:
        >>> search_space = {
        ...     "learning_rate": [0.001, 0.01, 0.1],
        ...     "batch_size": [32, 64, 128],
        ...     "hidden_dim": [64, 128, 256]
        ... }
        >>> search = GridSearch(search_space, maximize=True)
        >>> results = search.run(train_and_evaluate)
        >>> print(results.best_params)
    """

    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        maximize: bool = True,
        name: Optional[str] = None
    ) -> None:
        super().__init__(search_space, maximize, name or "GridSearch")

        # Validate search space
        for key, values in search_space.items():
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"GridSearch requires lists of values. Got {type(values)} for '{key}'")

        # Generate all combinations
        self._param_names = list(search_space.keys())
        self._param_values = [search_space[k] for k in self._param_names]
        self._combinations: Iterator[tuple] = iter(itertools.product(*self._param_values))
        self._total = 1
        for values in self._param_values:
            self._total *= len(values)

    def _get_next_params(self) -> Optional[Dict[str, Any]]:
        """Get the next parameter combination to evaluate."""
        try:
            values = next(self._combinations)
            return dict(zip(self._param_names, values))
        except StopIteration:
            return None

    @property
    def total_trials(self) -> int:
        """Return the total number of trials to run."""
        return self._total
