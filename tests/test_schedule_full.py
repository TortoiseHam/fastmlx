"""Tests for learning rate schedules."""

import unittest
import math

from fastmlx.schedule import (
    cosine_decay,
    linear_decay,
    step_decay,
    exponential_decay,
    polynomial_decay,
    warmup_cosine_decay,
    one_cycle,
    EpochScheduler,
    RepeatScheduler,
    LambdaScheduler,
)


class TestCosineDecay(unittest.TestCase):
    """Tests for cosine decay schedule."""

    def test_initial_lr(self) -> None:
        """Test that initial LR is correct."""
        lr = cosine_decay(step=0, cycle_length=100, init_lr=0.1)
        self.assertAlmostEqual(lr, 0.1, places=5)

    def test_midpoint(self) -> None:
        """Test LR at midpoint of cycle."""
        lr = cosine_decay(step=50, cycle_length=100, init_lr=1.0)
        # At midpoint, cosine decay should give ~0.5
        self.assertAlmostEqual(lr, 0.5, places=1)

    def test_end_of_cycle(self) -> None:
        """Test LR at end of cycle."""
        lr = cosine_decay(step=100, cycle_length=100, init_lr=1.0, min_lr=0.0)
        self.assertAlmostEqual(lr, 0.0, places=5)

    def test_min_lr(self) -> None:
        """Test minimum LR bound."""
        lr = cosine_decay(step=100, cycle_length=100, init_lr=1.0, min_lr=0.1)
        self.assertGreaterEqual(lr, 0.1)


class TestLinearDecay(unittest.TestCase):
    """Tests for linear decay schedule."""

    def test_initial_lr(self) -> None:
        """Test initial learning rate."""
        lr = linear_decay(step=0, total_steps=100, init_lr=0.1, end_lr=0.01)
        self.assertAlmostEqual(lr, 0.1, places=5)

    def test_final_lr(self) -> None:
        """Test final learning rate."""
        lr = linear_decay(step=100, total_steps=100, init_lr=0.1, end_lr=0.01)
        self.assertAlmostEqual(lr, 0.01, places=5)

    def test_linear_interpolation(self) -> None:
        """Test that decay is linear."""
        lr_25 = linear_decay(step=25, total_steps=100, init_lr=1.0, end_lr=0.0)
        lr_50 = linear_decay(step=50, total_steps=100, init_lr=1.0, end_lr=0.0)
        lr_75 = linear_decay(step=75, total_steps=100, init_lr=1.0, end_lr=0.0)

        self.assertAlmostEqual(lr_25, 0.75, places=5)
        self.assertAlmostEqual(lr_50, 0.50, places=5)
        self.assertAlmostEqual(lr_75, 0.25, places=5)


class TestStepDecay(unittest.TestCase):
    """Tests for step decay schedule."""

    def test_initial_lr(self) -> None:
        """Test initial learning rate."""
        lr = step_decay(step=0, init_lr=0.1, decay_factor=0.1, decay_steps=30)
        self.assertAlmostEqual(lr, 0.1, places=5)

    def test_after_first_decay(self) -> None:
        """Test LR after first decay step."""
        lr = step_decay(step=30, init_lr=0.1, decay_factor=0.1, decay_steps=30)
        self.assertAlmostEqual(lr, 0.01, places=5)

    def test_multiple_decays(self) -> None:
        """Test multiple decay steps."""
        lr = step_decay(step=60, init_lr=0.1, decay_factor=0.1, decay_steps=30)
        self.assertAlmostEqual(lr, 0.001, places=6)


class TestExponentialDecay(unittest.TestCase):
    """Tests for exponential decay schedule."""

    def test_initial_lr(self) -> None:
        """Test initial learning rate."""
        lr = exponential_decay(step=0, init_lr=0.1, decay_rate=0.96, decay_steps=100)
        self.assertAlmostEqual(lr, 0.1, places=5)

    def test_decay_rate(self) -> None:
        """Test decay rate application."""
        lr = exponential_decay(step=100, init_lr=0.1, decay_rate=0.96, decay_steps=100)
        self.assertAlmostEqual(lr, 0.096, places=4)


class TestPolynomialDecay(unittest.TestCase):
    """Tests for polynomial decay schedule."""

    def test_linear_decay(self) -> None:
        """Test polynomial decay with power=1 (linear)."""
        lr = polynomial_decay(step=50, total_steps=100, init_lr=1.0, end_lr=0.0, power=1.0)
        self.assertAlmostEqual(lr, 0.5, places=5)

    def test_quadratic_decay(self) -> None:
        """Test polynomial decay with power=2."""
        lr = polynomial_decay(step=50, total_steps=100, init_lr=1.0, end_lr=0.0, power=2.0)
        # (1 - 0.5)^2 * 1.0 = 0.25
        self.assertAlmostEqual(lr, 0.25, places=5)


class TestWarmupCosineDecay(unittest.TestCase):
    """Tests for warmup + cosine decay schedule."""

    def test_warmup_start(self) -> None:
        """Test LR at start of warmup."""
        lr = warmup_cosine_decay(
            step=0, warmup_steps=10, total_steps=100, init_lr=0.1
        )
        self.assertAlmostEqual(lr, 0.0, places=5)

    def test_warmup_end(self) -> None:
        """Test LR at end of warmup."""
        lr = warmup_cosine_decay(
            step=10, warmup_steps=10, total_steps=100, init_lr=0.1
        )
        self.assertAlmostEqual(lr, 0.1, places=5)

    def test_during_decay(self) -> None:
        """Test LR during cosine decay phase."""
        lr = warmup_cosine_decay(
            step=55, warmup_steps=10, total_steps=100, init_lr=0.1
        )
        # Should be between 0 and 0.1
        self.assertGreater(lr, 0)
        self.assertLess(lr, 0.1)


class TestOneCycle(unittest.TestCase):
    """Tests for one cycle schedule."""

    def test_cycle_start(self) -> None:
        """Test LR at start."""
        lr = one_cycle(step=0, total_steps=100, max_lr=0.1, div_factor=25)
        self.assertAlmostEqual(lr, 0.1 / 25, places=5)

    def test_cycle_peak(self) -> None:
        """Test LR at peak (30% of cycle)."""
        lr = one_cycle(step=30, total_steps=100, max_lr=0.1, pct_start=0.3)
        self.assertAlmostEqual(lr, 0.1, places=2)

    def test_cycle_end(self) -> None:
        """Test LR at end."""
        lr = one_cycle(
            step=100, total_steps=100, max_lr=0.1, div_factor=25, final_div_factor=1e4
        )
        # Should be very small
        self.assertLess(lr, 0.001)


class TestEpochScheduler(unittest.TestCase):
    """Tests for EpochScheduler."""

    def test_epoch_values(self) -> None:
        """Test getting values by epoch."""
        scheduler = EpochScheduler({0: 0.1, 10: 0.01, 20: 0.001})

        self.assertEqual(scheduler.get_value(epoch=0), 0.1)
        self.assertEqual(scheduler.get_value(epoch=5), 0.1)
        self.assertEqual(scheduler.get_value(epoch=10), 0.01)
        self.assertEqual(scheduler.get_value(epoch=15), 0.01)
        self.assertEqual(scheduler.get_value(epoch=25), 0.001)


class TestRepeatScheduler(unittest.TestCase):
    """Tests for RepeatScheduler."""

    def test_repeat_values(self) -> None:
        """Test repeating values."""
        scheduler = RepeatScheduler([0.1, 0.01, 0.001])

        self.assertEqual(scheduler.get_value(epoch=0), 0.1)
        self.assertEqual(scheduler.get_value(epoch=1), 0.01)
        self.assertEqual(scheduler.get_value(epoch=2), 0.001)
        self.assertEqual(scheduler.get_value(epoch=3), 0.1)  # Repeats
        self.assertEqual(scheduler.get_value(epoch=4), 0.01)


class TestLambdaScheduler(unittest.TestCase):
    """Tests for LambdaScheduler."""

    def test_custom_function(self) -> None:
        """Test custom schedule function."""
        scheduler = LambdaScheduler(lambda epoch: 0.1 * (0.9 ** epoch))

        self.assertAlmostEqual(scheduler.get_value(epoch=0), 0.1, places=5)
        self.assertAlmostEqual(scheduler.get_value(epoch=1), 0.09, places=5)
        self.assertAlmostEqual(scheduler.get_value(epoch=2), 0.081, places=5)


if __name__ == "__main__":
    unittest.main()
