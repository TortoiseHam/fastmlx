"""Tests for loss ops."""

import unittest

import mlx.core as mx

from fastmlx.op import (
    CrossEntropy,
    MeanSquaredError,
    L1Loss,
    FocalLoss,
    DiceLoss,
    HingeLoss,
    SmoothL1Loss,
)


class TestCrossEntropy(unittest.TestCase):
    """Tests for CrossEntropy loss."""

    def test_basic_loss(self) -> None:
        """Test basic cross entropy computation."""
        op = CrossEntropy(["y_pred", "y"], "loss")
        y_pred = mx.array([[0.9, 0.1], [0.2, 0.8]])  # logits
        y_true = mx.array([0, 1])  # class indices

        loss = op.forward([y_pred, y_true], {})

        self.assertIsInstance(loss, mx.array)
        self.assertEqual(loss.shape, ())
        # Loss should be positive
        self.assertGreater(float(loss.item()), 0)

    def test_perfect_prediction(self) -> None:
        """Test loss with near-perfect predictions."""
        op = CrossEntropy(["y_pred", "y"], "loss")
        # High confidence correct predictions (as logits)
        y_pred = mx.array([[10.0, -10.0], [-10.0, 10.0]])
        y_true = mx.array([0, 1])

        loss = op.forward([y_pred, y_true], {})

        # Loss should be very small
        self.assertLess(float(loss.item()), 0.1)


class TestMeanSquaredError(unittest.TestCase):
    """Tests for MeanSquaredError loss."""

    def test_basic_mse(self) -> None:
        """Test basic MSE computation."""
        op = MeanSquaredError(["y_pred", "y"], "loss")
        y_pred = mx.array([1.0, 2.0, 3.0])
        y_true = mx.array([1.0, 2.0, 3.0])

        loss = op.forward([y_pred, y_true], {})

        self.assertAlmostEqual(float(loss.item()), 0.0, places=5)

    def test_mse_with_error(self) -> None:
        """Test MSE with prediction errors."""
        op = MeanSquaredError(["y_pred", "y"], "loss")
        y_pred = mx.array([0.0, 0.0, 0.0])
        y_true = mx.array([1.0, 2.0, 3.0])

        loss = op.forward([y_pred, y_true], {})

        # MSE = (1^2 + 2^2 + 3^2) / 3 = 14/3 ≈ 4.667
        expected = (1 + 4 + 9) / 3
        self.assertAlmostEqual(float(loss.item()), expected, places=4)


class TestL1Loss(unittest.TestCase):
    """Tests for L1Loss."""

    def test_basic_l1(self) -> None:
        """Test basic L1 loss computation."""
        op = L1Loss(["y_pred", "y"], "loss")
        y_pred = mx.array([1.0, 2.0, 3.0])
        y_true = mx.array([1.0, 2.0, 3.0])

        loss = op.forward([y_pred, y_true], {})

        self.assertAlmostEqual(float(loss.item()), 0.0, places=5)

    def test_l1_with_error(self) -> None:
        """Test L1 with prediction errors."""
        op = L1Loss(["y_pred", "y"], "loss")
        y_pred = mx.array([0.0, 0.0, 0.0])
        y_true = mx.array([1.0, 2.0, 3.0])

        loss = op.forward([y_pred, y_true], {})

        # L1 = (|1| + |2| + |3|) / 3 = 6/3 = 2
        self.assertAlmostEqual(float(loss.item()), 2.0, places=4)


class TestFocalLoss(unittest.TestCase):
    """Tests for FocalLoss."""

    def test_focal_loss_basic(self) -> None:
        """Test basic focal loss computation."""
        op = FocalLoss(["y_pred", "y"], "loss", gamma=2.0, alpha=0.25)
        y_pred = mx.array([[0.9, 0.1], [0.2, 0.8]])
        y_true = mx.array([0, 1])

        loss = op.forward([y_pred, y_true], {})

        self.assertIsInstance(loss, mx.array)
        self.assertGreater(float(loss.item()), 0)

    def test_focal_loss_reduces_easy_examples(self) -> None:
        """Test that focal loss down-weights easy examples."""
        ce_op = CrossEntropy(["y_pred", "y"], "loss")
        focal_op = FocalLoss(["y_pred", "y"], "loss", gamma=2.0)

        # Easy example (high confidence correct prediction)
        y_pred = mx.array([[10.0, -10.0]])
        y_true = mx.array([0])

        ce_loss = ce_op.forward([y_pred, y_true], {})
        focal_loss = focal_op.forward([y_pred, y_true], {})

        # Focal loss should be smaller for easy examples
        self.assertLessEqual(float(focal_loss.item()), float(ce_loss.item()))


class TestDiceLoss(unittest.TestCase):
    """Tests for DiceLoss."""

    def test_perfect_overlap(self) -> None:
        """Test dice loss with perfect overlap."""
        op = DiceLoss(["y_pred", "y"], "loss", smooth=1e-6)
        y_pred = mx.array([[1.0, 1.0], [0.0, 0.0]])
        y_true = mx.array([[1.0, 1.0], [0.0, 0.0]])

        loss = op.forward([y_pred, y_true], {})

        # Perfect overlap should give loss close to 0
        self.assertLess(float(loss.item()), 0.01)

    def test_no_overlap(self) -> None:
        """Test dice loss with no overlap."""
        op = DiceLoss(["y_pred", "y"], "loss", smooth=1e-6)
        y_pred = mx.array([[1.0, 1.0], [0.0, 0.0]])
        y_true = mx.array([[0.0, 0.0], [1.0, 1.0]])

        loss = op.forward([y_pred, y_true], {})

        # No overlap should give loss close to 1
        self.assertGreater(float(loss.item()), 0.9)


class TestHingeLoss(unittest.TestCase):
    """Tests for HingeLoss."""

    def test_correct_classification(self) -> None:
        """Test hinge loss with correct classifications."""
        op = HingeLoss(["y_pred", "y"], "loss")
        # Predictions > 1 for positive class, < -1 for negative
        y_pred = mx.array([2.0, -2.0])
        y_true = mx.array([1.0, -1.0])

        loss = op.forward([y_pred, y_true], {})

        # Correct with margin, loss should be 0
        self.assertAlmostEqual(float(loss.item()), 0.0, places=5)

    def test_incorrect_classification(self) -> None:
        """Test hinge loss with incorrect classifications."""
        op = HingeLoss(["y_pred", "y"], "loss")
        y_pred = mx.array([-1.0])  # Wrong sign
        y_true = mx.array([1.0])

        loss = op.forward([y_pred, y_true], {})

        # max(0, 1 - (-1)*1) = max(0, 2) = 2
        self.assertAlmostEqual(float(loss.item()), 2.0, places=4)


class TestSmoothL1Loss(unittest.TestCase):
    """Tests for SmoothL1Loss."""

    def test_small_errors_quadratic(self) -> None:
        """Test that small errors use quadratic penalty."""
        op = SmoothL1Loss(["y_pred", "y"], "loss", beta=1.0)
        y_pred = mx.array([0.5])
        y_true = mx.array([0.0])

        loss = op.forward([y_pred, y_true], {})

        # For |x| < beta: 0.5 * x^2 / beta = 0.5 * 0.25 / 1 = 0.125
        self.assertAlmostEqual(float(loss.item()), 0.125, places=4)

    def test_large_errors_linear(self) -> None:
        """Test that large errors use linear penalty."""
        op = SmoothL1Loss(["y_pred", "y"], "loss", beta=1.0)
        y_pred = mx.array([2.0])
        y_true = mx.array([0.0])

        loss = op.forward([y_pred, y_true], {})

        # For |x| >= beta: |x| - 0.5 * beta = 2 - 0.5 = 1.5
        self.assertAlmostEqual(float(loss.item()), 1.5, places=4)


if __name__ == "__main__":
    unittest.main()
