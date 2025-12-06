"""Tests for trace classes."""

import unittest
import tempfile
import os

import mlx.core as mx

from fastmlx.trace import (
    Accuracy,
    LossMonitor,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    MCC,
    Dice,
    EarlyStopping,
    ReduceLROnPlateau,
    TerminateOnNaN,
)


class TestAccuracy(unittest.TestCase):
    """Tests for Accuracy trace."""

    def test_perfect_accuracy(self) -> None:
        """Test accuracy with perfect predictions."""
        trace = Accuracy(true_key="y", pred_key="y_pred")
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        # Perfect predictions
        batch = {
            "y": mx.array([0, 1, 2, 0, 1]),
            "y_pred": mx.array([
                [1.0, 0.0, 0.0],  # class 0
                [0.0, 1.0, 0.0],  # class 1
                [0.0, 0.0, 1.0],  # class 2
                [1.0, 0.0, 0.0],  # class 0
                [0.0, 1.0, 0.0],  # class 1
            ])
        }
        trace.on_batch_end(batch, state)
        trace.on_epoch_end(state)

        self.assertAlmostEqual(state["metrics"]["accuracy"], 1.0, places=5)

    def test_half_accuracy(self) -> None:
        """Test 50% accuracy."""
        trace = Accuracy()
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        batch = {
            "y": mx.array([0, 1, 0, 1]),
            "y_pred": mx.array([
                [1.0, 0.0],  # correct
                [1.0, 0.0],  # wrong
                [1.0, 0.0],  # correct
                [1.0, 0.0],  # wrong
            ])
        }
        trace.on_batch_end(batch, state)
        trace.on_epoch_end(state)

        self.assertAlmostEqual(state["metrics"]["accuracy"], 0.5, places=5)

    def test_one_hot_labels(self) -> None:
        """Test accuracy with one-hot encoded labels."""
        trace = Accuracy()
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        batch = {
            "y": mx.array([[1, 0], [0, 1]]),  # one-hot
            "y_pred": mx.array([[1.0, 0.0], [0.0, 1.0]]),
        }
        trace.on_batch_end(batch, state)
        trace.on_epoch_end(state)

        self.assertAlmostEqual(state["metrics"]["accuracy"], 1.0, places=5)


class TestLossMonitor(unittest.TestCase):
    """Tests for LossMonitor trace."""

    def test_loss_tracking(self) -> None:
        """Test loss value tracking."""
        trace = LossMonitor(loss_key="ce")
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        trace.on_batch_end({"ce": mx.array(1.0)}, state)
        trace.on_batch_end({"ce": mx.array(2.0)}, state)
        trace.on_batch_end({"ce": mx.array(3.0)}, state)

        trace.on_epoch_end(state)

        self.assertAlmostEqual(state["metrics"]["ce"], 2.0, places=5)  # mean


class TestPrecision(unittest.TestCase):
    """Tests for Precision trace."""

    def test_precision_computation(self) -> None:
        """Test precision computation."""
        trace = Precision(num_classes=2, average="macro")
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        batch = {
            "y": mx.array([0, 0, 1, 1]),
            "y_pred": mx.array([
                [1.0, 0.0],  # TP for class 0
                [1.0, 0.0],  # TP for class 0
                [1.0, 0.0],  # FP for class 0, FN for class 1
                [0.0, 1.0],  # TP for class 1
            ])
        }
        trace.on_batch_end(batch, state)
        trace.on_epoch_end(state)

        # Class 0: TP=2, FP=1, precision=2/3
        # Class 1: TP=1, FP=0, precision=1
        # Macro: (2/3 + 1) / 2 = 5/6
        self.assertGreater(state["metrics"]["precision"], 0.8)


class TestRecall(unittest.TestCase):
    """Tests for Recall trace."""

    def test_recall_computation(self) -> None:
        """Test recall computation."""
        trace = Recall(num_classes=2, average="macro")
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        batch = {
            "y": mx.array([0, 0, 1, 1]),
            "y_pred": mx.array([
                [1.0, 0.0],  # TP for class 0
                [0.0, 1.0],  # FN for class 0
                [0.0, 1.0],  # TP for class 1
                [0.0, 1.0],  # TP for class 1
            ])
        }
        trace.on_batch_end(batch, state)
        trace.on_epoch_end(state)

        # Class 0: TP=1, FN=1, recall=0.5
        # Class 1: TP=2, FN=0, recall=1
        # Macro: (0.5 + 1) / 2 = 0.75
        self.assertAlmostEqual(state["metrics"]["recall"], 0.75, places=2)


class TestF1Score(unittest.TestCase):
    """Tests for F1Score trace."""

    def test_f1_perfect(self) -> None:
        """Test F1 with perfect predictions."""
        trace = F1Score(num_classes=2)
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        batch = {
            "y": mx.array([0, 0, 1, 1]),
            "y_pred": mx.array([
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ])
        }
        trace.on_batch_end(batch, state)
        trace.on_epoch_end(state)

        self.assertAlmostEqual(state["metrics"]["f1_score"], 1.0, places=5)


class TestConfusionMatrix(unittest.TestCase):
    """Tests for ConfusionMatrix trace."""

    def test_confusion_matrix(self) -> None:
        """Test confusion matrix computation."""
        trace = ConfusionMatrix(num_classes=3)
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        batch = {
            "y": mx.array([0, 1, 2, 0]),
            "y_pred": mx.array([
                [1.0, 0.0, 0.0],  # correct
                [0.0, 1.0, 0.0],  # correct
                [0.0, 0.0, 1.0],  # correct
                [0.0, 1.0, 0.0],  # predicted 1, actual 0
            ])
        }
        trace.on_batch_end(batch, state)
        trace.on_epoch_end(state)

        cm = state["metrics"]["confusion_matrix"]
        self.assertEqual(len(cm), 3)
        self.assertEqual(cm[0][0], 1)  # True class 0, predicted 0
        self.assertEqual(cm[0][1], 1)  # True class 0, predicted 1
        self.assertEqual(cm[1][1], 1)  # True class 1, predicted 1
        self.assertEqual(cm[2][2], 1)  # True class 2, predicted 2


class TestMCC(unittest.TestCase):
    """Tests for MCC trace."""

    def test_mcc_perfect(self) -> None:
        """Test MCC with perfect predictions."""
        trace = MCC(num_classes=2)
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        batch = {
            "y": mx.array([0, 0, 1, 1]),
            "y_pred": mx.array([
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ])
        }
        trace.on_batch_end(batch, state)
        trace.on_epoch_end(state)

        self.assertAlmostEqual(state["metrics"]["mcc"], 1.0, places=5)

    def test_mcc_random(self) -> None:
        """Test MCC with random predictions."""
        trace = MCC(num_classes=2)
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        # Alternating predictions regardless of true label
        batch = {
            "y": mx.array([0, 0, 1, 1]),
            "y_pred": mx.array([
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ])
        }
        trace.on_batch_end(batch, state)
        trace.on_epoch_end(state)

        # Should be close to 0 for random-like predictions
        self.assertLess(abs(state["metrics"]["mcc"]), 0.5)


class TestDice(unittest.TestCase):
    """Tests for Dice trace."""

    def test_dice_perfect(self) -> None:
        """Test Dice with perfect segmentation."""
        trace = Dice(threshold=0.5)
        state = {"metrics": {}}

        trace.on_epoch_begin(state)

        batch = {
            "y": mx.array([[1.0, 1.0], [0.0, 0.0]]),
            "y_pred": mx.array([[0.9, 0.9], [0.1, 0.1]]),
        }
        trace.on_batch_end(batch, state)
        trace.on_epoch_end(state)

        self.assertGreater(state["metrics"]["dice"], 0.99)


class TestEarlyStopping(unittest.TestCase):
    """Tests for EarlyStopping trace."""

    def test_early_stopping_triggered(self) -> None:
        """Test that early stopping is triggered after patience epochs."""
        trace = EarlyStopping(monitor="loss", patience=2, mode="min")
        state = {"metrics": {}, "should_stop": False}

        # Loss not improving
        trace.on_epoch_begin(state)
        state["metrics"]["loss"] = 1.0
        trace.on_epoch_end(state)
        self.assertFalse(state.get("should_stop", False))

        trace.on_epoch_begin(state)
        state["metrics"]["loss"] = 1.1  # worse
        trace.on_epoch_end(state)
        self.assertFalse(state.get("should_stop", False))

        trace.on_epoch_begin(state)
        state["metrics"]["loss"] = 1.2  # worse again
        trace.on_epoch_end(state)
        self.assertTrue(state.get("should_stop", False))

    def test_early_stopping_reset(self) -> None:
        """Test that counter resets on improvement."""
        trace = EarlyStopping(monitor="acc", patience=2, mode="max")
        state = {"metrics": {}, "should_stop": False}

        trace.on_epoch_begin(state)
        state["metrics"]["acc"] = 0.5
        trace.on_epoch_end(state)

        trace.on_epoch_begin(state)
        state["metrics"]["acc"] = 0.4  # worse
        trace.on_epoch_end(state)

        trace.on_epoch_begin(state)
        state["metrics"]["acc"] = 0.6  # better - should reset
        trace.on_epoch_end(state)

        trace.on_epoch_begin(state)
        state["metrics"]["acc"] = 0.55  # worse
        trace.on_epoch_end(state)

        self.assertFalse(state.get("should_stop", False))


class TestTerminateOnNaN(unittest.TestCase):
    """Tests for TerminateOnNaN trace."""

    def test_detects_nan(self) -> None:
        """Test that NaN is detected."""
        trace = TerminateOnNaN(monitor="loss")
        state = {"metrics": {}, "should_stop": False}

        trace.on_epoch_begin(state)
        state["metrics"]["loss"] = float("nan")
        trace.on_epoch_end(state)

        self.assertTrue(state.get("should_stop", False))

    def test_no_nan(self) -> None:
        """Test normal operation without NaN."""
        trace = TerminateOnNaN(monitor="loss")
        state = {"metrics": {}, "should_stop": False}

        trace.on_epoch_begin(state)
        state["metrics"]["loss"] = 0.5
        trace.on_epoch_end(state)

        self.assertFalse(state.get("should_stop", False))


if __name__ == "__main__":
    unittest.main()
