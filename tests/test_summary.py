"""Tests for summary and experiment tracking module."""

import unittest
import tempfile
import os
import json

from fastmlx.summary import Summary, Experiment


class TestSummary(unittest.TestCase):
    """Tests for Summary class."""

    def test_add_metric(self) -> None:
        """Test adding metrics."""
        summary = Summary()

        summary.add("loss", 1.0, step=0)
        summary.add("loss", 0.5, step=1)
        summary.add("loss", 0.25, step=2)

        self.assertEqual(len(summary.get("loss")), 3)

    def test_get_metric(self) -> None:
        """Test getting metric history."""
        summary = Summary()
        summary.add("accuracy", 0.5, step=0)
        summary.add("accuracy", 0.7, step=1)
        summary.add("accuracy", 0.9, step=2)

        history = summary.get("accuracy")

        self.assertEqual(history, [0.5, 0.7, 0.9])

    def test_get_with_steps(self) -> None:
        """Test getting metrics with step numbers."""
        summary = Summary()
        summary.add("loss", 1.0, step=0)
        summary.add("loss", 0.5, step=10)

        history = summary.get("loss", with_steps=True)

        self.assertEqual(history, [(0, 1.0), (10, 0.5)])

    def test_best_metric(self) -> None:
        """Test finding best metric value."""
        summary = Summary()
        summary.add("val_acc", 0.5, step=0)
        summary.add("val_acc", 0.9, step=1)
        summary.add("val_acc", 0.7, step=2)

        best_step, best_val = summary.best("val_acc", mode="max")

        self.assertEqual(best_step, 1)
        self.assertEqual(best_val, 0.9)

    def test_best_metric_min(self) -> None:
        """Test finding minimum metric value."""
        summary = Summary()
        summary.add("loss", 1.0, step=0)
        summary.add("loss", 0.1, step=1)
        summary.add("loss", 0.5, step=2)

        best_step, best_val = summary.best("loss", mode="min")

        self.assertEqual(best_step, 1)
        self.assertEqual(best_val, 0.1)

    def test_describe(self) -> None:
        """Test summary statistics."""
        summary = Summary()
        for i in range(10):
            summary.add("metric", float(i), step=i)

        stats = summary.describe("metric")

        self.assertEqual(stats["count"], 10)
        self.assertAlmostEqual(stats["mean"], 4.5, places=5)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 9.0)

    def test_multiple_metrics(self) -> None:
        """Test tracking multiple metrics."""
        summary = Summary()
        summary.add("loss", 1.0, step=0)
        summary.add("accuracy", 0.5, step=0)
        summary.add("loss", 0.5, step=1)
        summary.add("accuracy", 0.8, step=1)

        self.assertEqual(len(summary.get("loss")), 2)
        self.assertEqual(len(summary.get("accuracy")), 2)
        self.assertEqual(len(summary.metrics), 2)

    def test_clear(self) -> None:
        """Test clearing summary."""
        summary = Summary()
        summary.add("loss", 1.0, step=0)
        summary.add("accuracy", 0.5, step=0)

        summary.clear()

        self.assertEqual(len(summary.metrics), 0)


class TestExperiment(unittest.TestCase):
    """Tests for Experiment class."""

    def test_experiment_creation(self) -> None:
        """Test creating an experiment."""
        exp = Experiment(name="test_exp")

        self.assertEqual(exp.name, "test_exp")
        self.assertIsNotNone(exp.start_time)

    def test_log_config(self) -> None:
        """Test logging configuration."""
        exp = Experiment(name="test")
        exp.log_config({
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 10
        })

        self.assertEqual(exp.config["learning_rate"], 0.01)
        self.assertEqual(exp.config["batch_size"], 32)

    def test_log_metric(self) -> None:
        """Test logging metrics."""
        exp = Experiment(name="test")

        exp.log("loss", 1.0, step=0)
        exp.log("loss", 0.5, step=1)

        self.assertEqual(len(exp.summary.get("loss")), 2)

    def test_experiment_duration(self) -> None:
        """Test experiment duration tracking."""
        exp = Experiment(name="test")

        import time
        time.sleep(0.1)

        exp.finish()

        self.assertGreater(exp.duration, 0)

    def test_save_and_load(self) -> None:
        """Test saving and loading experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save experiment
            exp = Experiment(name="save_test")
            exp.log_config({"lr": 0.01})
            exp.log("loss", 1.0, step=0)
            exp.log("loss", 0.5, step=1)
            exp.finish()

            save_path = os.path.join(tmpdir, "experiment.json")
            exp.save(save_path)

            # Load experiment
            loaded = Experiment.load(save_path)

            self.assertEqual(loaded.name, "save_test")
            self.assertEqual(loaded.config["lr"], 0.01)
            self.assertEqual(len(loaded.summary.get("loss")), 2)

    def test_experiment_tags(self) -> None:
        """Test experiment tags."""
        exp = Experiment(name="test", tags=["baseline", "v1"])

        self.assertIn("baseline", exp.tags)
        self.assertIn("v1", exp.tags)

    def test_experiment_notes(self) -> None:
        """Test experiment notes."""
        exp = Experiment(name="test")
        exp.add_note("Testing learning rate 0.01")
        exp.add_note("Model converged after 50 epochs")

        self.assertEqual(len(exp.notes), 2)
        self.assertIn("Testing learning rate", exp.notes[0])

    def test_system_info(self) -> None:
        """Test system info logging."""
        exp = Experiment(name="test", log_system_info=True)

        self.assertIn("platform", exp.system_info)
        self.assertIn("python_version", exp.system_info)


class TestSummaryExport(unittest.TestCase):
    """Tests for exporting summary data."""

    def test_to_dict(self) -> None:
        """Test exporting summary to dict."""
        summary = Summary()
        summary.add("loss", 1.0, step=0)
        summary.add("loss", 0.5, step=1)
        summary.add("acc", 0.9, step=1)

        data = summary.to_dict()

        self.assertIn("loss", data)
        self.assertIn("acc", data)
        self.assertEqual(len(data["loss"]), 2)

    def test_to_csv(self) -> None:
        """Test exporting summary to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Summary()
            summary.add("loss", 1.0, step=0)
            summary.add("loss", 0.5, step=1)

            csv_path = os.path.join(tmpdir, "metrics.csv")
            summary.to_csv(csv_path)

            self.assertTrue(os.path.exists(csv_path))

            # Verify CSV contents
            with open(csv_path, "r") as f:
                contents = f.read()
                self.assertIn("loss", contents)
                self.assertIn("1.0", contents)


if __name__ == "__main__":
    unittest.main()
