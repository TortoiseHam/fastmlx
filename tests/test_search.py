"""Tests for hyperparameter search module."""

import unittest

from fastmlx.search import (
    GridSearch,
    RandomSearch,
    GoldenSection,
    SearchResults,
)


class TestGridSearch(unittest.TestCase):
    """Tests for GridSearch."""

    def test_grid_search_basic(self) -> None:
        """Test basic grid search."""
        def objective(lr, batch_size):
            # Simple function: minimize difference from targets
            return abs(lr - 0.01) + abs(batch_size - 32)

        search = GridSearch(
            objective_fn=objective,
            params={
                "lr": [0.001, 0.01, 0.1],
                "batch_size": [16, 32, 64],
            }
        )

        results = search.run()

        self.assertIsInstance(results, SearchResults)
        self.assertEqual(results.best_params["lr"], 0.01)
        self.assertEqual(results.best_params["batch_size"], 32)
        self.assertEqual(len(results.all_results), 9)  # 3 x 3 grid

    def test_grid_search_maximize(self) -> None:
        """Test grid search with maximization."""
        def objective(x):
            return -x ** 2  # Maximum at x=0

        search = GridSearch(
            objective_fn=objective,
            params={"x": [-2, -1, 0, 1, 2]},
            maximize=True
        )

        results = search.run()

        self.assertEqual(results.best_params["x"], 0)

    def test_grid_search_single_param(self) -> None:
        """Test grid search with single parameter."""
        def objective(x):
            return (x - 5) ** 2

        search = GridSearch(
            objective_fn=objective,
            params={"x": [0, 5, 10]}
        )

        results = search.run()

        self.assertEqual(results.best_params["x"], 5)
        self.assertAlmostEqual(results.best_score, 0.0, places=5)


class TestRandomSearch(unittest.TestCase):
    """Tests for RandomSearch."""

    def test_random_search_basic(self) -> None:
        """Test basic random search."""
        def objective(x, y):
            return (x - 1) ** 2 + (y - 2) ** 2

        search = RandomSearch(
            objective_fn=objective,
            params={
                "x": {"type": "uniform", "low": 0, "high": 2},
                "y": {"type": "uniform", "low": 1, "high": 3},
            },
            n_iter=50,
            seed=42
        )

        results = search.run()

        # Should find values close to optimum
        self.assertLess(abs(results.best_params["x"] - 1), 0.5)
        self.assertLess(abs(results.best_params["y"] - 2), 0.5)

    def test_random_search_choice(self) -> None:
        """Test random search with choice distribution."""
        def objective(model_type):
            scores = {"small": 0.8, "medium": 0.9, "large": 0.85}
            return scores[model_type]

        search = RandomSearch(
            objective_fn=objective,
            params={
                "model_type": {"type": "choice", "values": ["small", "medium", "large"]}
            },
            n_iter=30,
            maximize=True,
            seed=42
        )

        results = search.run()

        self.assertEqual(results.best_params["model_type"], "medium")

    def test_random_search_log_uniform(self) -> None:
        """Test random search with log-uniform distribution."""
        def objective(lr):
            # Optimum at lr=0.01
            return abs(lr - 0.01)

        search = RandomSearch(
            objective_fn=objective,
            params={
                "lr": {"type": "log_uniform", "low": 1e-4, "high": 1e-1}
            },
            n_iter=100,
            seed=42
        )

        results = search.run()

        # Should find value in the right order of magnitude
        self.assertLess(results.best_params["lr"], 0.1)
        self.assertGreater(results.best_params["lr"], 0.001)


class TestGoldenSection(unittest.TestCase):
    """Tests for GoldenSection search."""

    def test_golden_section_basic(self) -> None:
        """Test basic golden section search."""
        def objective(x):
            return (x - 3) ** 2  # Minimum at x=3

        search = GoldenSection(
            objective_fn=objective,
            param_name="x",
            low=0,
            high=10,
            tol=1e-5
        )

        results = search.run()

        self.assertAlmostEqual(results.best_params["x"], 3.0, places=4)

    def test_golden_section_maximize(self) -> None:
        """Test golden section with maximization."""
        def objective(x):
            return -(x - 5) ** 2  # Maximum at x=5

        search = GoldenSection(
            objective_fn=objective,
            param_name="x",
            low=0,
            high=10,
            maximize=True,
            tol=1e-5
        )

        results = search.run()

        self.assertAlmostEqual(results.best_params["x"], 5.0, places=4)

    def test_golden_section_convergence(self) -> None:
        """Test that golden section converges."""
        call_count = [0]

        def objective(x):
            call_count[0] += 1
            return x ** 2

        search = GoldenSection(
            objective_fn=objective,
            param_name="x",
            low=-10,
            high=10,
            tol=1e-3
        )

        results = search.run()

        # Should find minimum near 0
        self.assertLess(abs(results.best_params["x"]), 0.01)
        # Should converge in reasonable number of iterations
        self.assertLess(call_count[0], 50)


class TestSearchResults(unittest.TestCase):
    """Tests for SearchResults dataclass."""

    def test_search_results_creation(self) -> None:
        """Test creating SearchResults."""
        results = SearchResults(
            best_params={"x": 1, "y": 2},
            best_score=0.95,
            all_results=[
                ({"x": 0, "y": 0}, 0.5),
                ({"x": 1, "y": 2}, 0.95),
            ]
        )

        self.assertEqual(results.best_params["x"], 1)
        self.assertEqual(results.best_score, 0.95)
        self.assertEqual(len(results.all_results), 2)

    def test_search_results_to_dataframe(self) -> None:
        """Test converting results to pandas DataFrame."""
        results = SearchResults(
            best_params={"lr": 0.01},
            best_score=0.9,
            all_results=[
                ({"lr": 0.1}, 0.7),
                ({"lr": 0.01}, 0.9),
                ({"lr": 0.001}, 0.8),
            ]
        )

        # If pandas is available, test conversion
        try:
            df = results.to_dataframe()
            self.assertEqual(len(df), 3)
            self.assertIn("lr", df.columns)
            self.assertIn("score", df.columns)
        except ImportError:
            pass  # pandas not installed


if __name__ == "__main__":
    unittest.main()
