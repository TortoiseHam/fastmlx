"""Integration tests for Estimator training loop."""

import unittest
from typing import Any, MutableMapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from fastmlx.estimator import Estimator
from fastmlx.pipeline import Pipeline
from fastmlx.network import Network
from fastmlx.op import ModelOp, UpdateOp, CrossEntropyLoss
from fastmlx.trace import EarlyStopping, TerminateOnNaN
from fastmlx.trace.base import Trace


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 10, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def __call__(self, x):
        return self.linear(x)


class SimpleDataset:
    """Simple dataset that yields a fixed number of batches."""

    def __init__(self, num_samples: int = 100, input_dim: int = 10, num_classes: int = 3):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = mx.random.normal((self.input_dim,))
        y = mx.array(idx % self.num_classes)
        return {"x": x, "y": y}


class CountingTrace(Trace):
    """Trace that counts how many epochs were run."""

    def __init__(self):
        self.epochs_started = 0
        self.epochs_ended = 0
        self.batches_processed = 0

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        if state.get("mode") == "train":
            self.epochs_started += 1

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        if state.get("mode") == "train":
            self.epochs_ended += 1

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        if state.get("mode") == "train":
            self.batches_processed += 1


class FakeMetricTrace(Trace):
    """Trace that sets a fake metric value each epoch."""

    def __init__(self, metric_name: str, values: list):
        self.metric_name = metric_name
        self.values = values
        self.epoch = 0

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        if state.get("mode") == "train":
            if self.epoch < len(self.values):
                state.setdefault("metrics", {})[self.metric_name] = self.values[self.epoch]
            self.epoch += 1


class NaNLossInjectTrace(Trace):
    """Trace that injects NaN into the loss after N batches."""

    def __init__(self, loss_key: str, inject_after: int):
        self.loss_key = loss_key
        self.inject_after = inject_after
        self.batch_count = 0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        self.batch_count += 1
        if self.batch_count >= self.inject_after:
            batch[self.loss_key] = float("nan")


class TestEstimatorBasic(unittest.TestCase):
    """Basic Estimator functionality tests."""

    def test_estimator_creation(self) -> None:
        """Test creating an Estimator."""
        dataset = SimpleDataset(num_samples=10)
        pipeline = Pipeline(train_data=dataset, batch_size=2)
        network = Network()

        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=1
        )

        self.assertEqual(estimator.epochs, 1)
        self.assertIsNotNone(estimator.pipeline)
        self.assertIsNotNone(estimator.network)

    def test_estimator_runs_all_epochs(self) -> None:
        """Test that Estimator runs all epochs when not stopped early."""
        dataset = SimpleDataset(num_samples=10)
        pipeline = Pipeline(train_data=dataset, batch_size=5)
        network = Network()

        counter = CountingTrace()
        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=3,
            traces=[counter]
        )

        estimator.fit()

        self.assertEqual(counter.epochs_started, 3)
        self.assertEqual(counter.epochs_ended, 3)


class TestEarlyStopping(unittest.TestCase):
    """Tests for EarlyStopping integration."""

    def test_early_stopping_stops_training(self) -> None:
        """Test that EarlyStopping actually stops training."""
        dataset = SimpleDataset(num_samples=10)
        pipeline = Pipeline(train_data=dataset, batch_size=5)
        network = Network()

        # Metrics that don't improve: 1.0, 1.0, 1.0, 1.0, 1.0
        # With patience=2, should stop at epoch 3 (after 2 non-improvements)
        fake_metrics = FakeMetricTrace("loss", [1.0, 1.0, 1.0, 1.0, 1.0])
        early_stopping = EarlyStopping(monitor="loss", patience=2, mode="min")
        counter = CountingTrace()

        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=10,  # Would run 10 epochs without early stopping
            traces=[fake_metrics, early_stopping, counter]
        )

        state = estimator.fit()

        # Should have stopped early - ran fewer than 10 epochs
        self.assertLess(counter.epochs_ended, 10)
        # Should have run at least 3 epochs (initial + patience)
        self.assertGreaterEqual(counter.epochs_ended, 3)

    def test_early_stopping_continues_when_improving(self) -> None:
        """Test that EarlyStopping doesn't stop when metric improves."""
        dataset = SimpleDataset(num_samples=10)
        pipeline = Pipeline(train_data=dataset, batch_size=5)
        network = Network()

        # Metrics that keep improving
        fake_metrics = FakeMetricTrace("loss", [1.0, 0.9, 0.8, 0.7, 0.6])
        early_stopping = EarlyStopping(monitor="loss", patience=2, mode="min")
        counter = CountingTrace()

        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=5,
            traces=[fake_metrics, early_stopping, counter]
        )

        estimator.fit()

        # Should have run all 5 epochs since loss kept improving
        self.assertEqual(counter.epochs_ended, 5)

    def test_early_stopping_max_mode(self) -> None:
        """Test EarlyStopping in max mode (e.g., accuracy)."""
        dataset = SimpleDataset(num_samples=10)
        pipeline = Pipeline(train_data=dataset, batch_size=5)
        network = Network()

        # Accuracy that plateaus: 0.8, 0.8, 0.8...
        fake_metrics = FakeMetricTrace("accuracy", [0.8, 0.8, 0.8, 0.8, 0.8])
        early_stopping = EarlyStopping(monitor="accuracy", patience=2, mode="max")
        counter = CountingTrace()

        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=10,
            traces=[fake_metrics, early_stopping, counter]
        )

        estimator.fit()

        # Should have stopped early
        self.assertLess(counter.epochs_ended, 10)


class TestTerminateOnNaN(unittest.TestCase):
    """Tests for TerminateOnNaN integration."""

    def test_terminate_on_nan_stops_training(self) -> None:
        """Test that TerminateOnNaN stops training when NaN is detected."""
        dataset = SimpleDataset(num_samples=20)
        pipeline = Pipeline(train_data=dataset, batch_size=2)
        network = Network()

        # Inject NaN after 3 batches
        nan_injector = NaNLossInjectTrace("loss", inject_after=3)
        terminate_on_nan = TerminateOnNaN(monitor="loss")
        counter = CountingTrace()

        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=10,
            traces=[nan_injector, terminate_on_nan, counter]
        )

        estimator.fit()

        # Should have processed very few batches before stopping
        self.assertLess(counter.batches_processed, 10)


class TestEstimatorState(unittest.TestCase):
    """Tests for Estimator state management."""

    def test_state_contains_mode(self) -> None:
        """Test that state contains mode information."""
        dataset = SimpleDataset(num_samples=10)
        pipeline = Pipeline(train_data=dataset, batch_size=5)
        network = Network()

        class ModeChecker(Trace):
            def __init__(self):
                self.modes_seen = []

            def on_epoch_begin(self, state):
                self.modes_seen.append(state.get("mode"))

        mode_checker = ModeChecker()
        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=2,
            traces=[mode_checker]
        )

        estimator.fit()

        self.assertIn("train", mode_checker.modes_seen)

    def test_state_contains_epoch(self) -> None:
        """Test that state contains epoch information."""
        dataset = SimpleDataset(num_samples=10)
        pipeline = Pipeline(train_data=dataset, batch_size=5)
        network = Network()

        class EpochChecker(Trace):
            def __init__(self):
                self.epochs_seen = []

            def on_epoch_begin(self, state):
                self.epochs_seen.append(state.get("epoch"))

        epoch_checker = EpochChecker()
        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=3,
            traces=[epoch_checker]
        )

        estimator.fit()

        # Epochs are 0-indexed in state
        self.assertEqual(epoch_checker.epochs_seen, [0, 1, 2])

    def test_should_stop_flag_respected(self) -> None:
        """Test that should_stop flag in state stops training."""
        dataset = SimpleDataset(num_samples=10)
        pipeline = Pipeline(train_data=dataset, batch_size=5)
        network = Network()

        class StopAfterOneEpoch(Trace):
            def on_epoch_end(self, state):
                state["should_stop"] = True

        stopper = StopAfterOneEpoch()
        counter = CountingTrace()
        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=10,
            traces=[stopper, counter]
        )

        estimator.fit()

        # Should have only run 1 epoch
        self.assertEqual(counter.epochs_ended, 1)


class TestEstimatorWithModel(unittest.TestCase):
    """Tests for Estimator with actual model training."""

    def test_training_with_simple_model(self) -> None:
        """Test training loop with a simple model."""
        mx.random.seed(42)

        # Create model
        model = SimpleModel(input_dim=10, num_classes=3)
        optimizer = optim.SGD(learning_rate=0.01)

        # Create dataset
        dataset = SimpleDataset(num_samples=20, input_dim=10, num_classes=3)
        pipeline = Pipeline(train_data=dataset, batch_size=4)

        # Create network with ops
        network = Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            CrossEntropyLoss(inputs=("y_pred", "y"), outputs="loss"),
            UpdateOp(model=model, optimizer=optimizer, loss="loss")
        ])

        counter = CountingTrace()
        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=2,
            traces=[counter],
            log_interval=100  # Reduce logging
        )

        state = estimator.fit()

        # Should have completed both epochs
        self.assertEqual(counter.epochs_ended, 2)
        self.assertGreater(counter.batches_processed, 0)


class TestEstimatorEvaluation(unittest.TestCase):
    """Tests for Estimator evaluation functionality."""

    def test_evaluation_runs_after_training(self) -> None:
        """Test that evaluation runs when eval_data is provided."""
        train_dataset = SimpleDataset(num_samples=10)
        eval_dataset = SimpleDataset(num_samples=5)

        pipeline = Pipeline(
            train_data=train_dataset,
            eval_data=eval_dataset,
            batch_size=5
        )
        network = Network()

        class EvalChecker(Trace):
            def __init__(self):
                self.eval_epochs = 0

            def on_epoch_end(self, state):
                if state.get("mode") == "eval":
                    self.eval_epochs += 1

        eval_checker = EvalChecker()
        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=2,
            traces=[eval_checker]
        )

        estimator.fit()

        # Should have run evaluation after each training epoch
        self.assertEqual(eval_checker.eval_epochs, 2)

    def test_test_method(self) -> None:
        """Test the test() method for evaluation only."""
        eval_dataset = SimpleDataset(num_samples=10)
        pipeline = Pipeline(
            train_data=SimpleDataset(num_samples=5),
            eval_data=eval_dataset,
            batch_size=5
        )
        network = Network()

        class BatchCounter(Trace):
            def __init__(self):
                self.batch_count = 0

            def on_batch_end(self, batch, state):
                self.batch_count += 1

        counter = BatchCounter()
        estimator = Estimator(
            pipeline=pipeline,
            network=network,
            epochs=1,
            traces=[counter]
        )

        state = estimator.test()

        # Should have processed eval batches
        self.assertGreater(counter.batch_count, 0)
        self.assertEqual(state.get("mode"), "eval")


if __name__ == "__main__":
    unittest.main()
