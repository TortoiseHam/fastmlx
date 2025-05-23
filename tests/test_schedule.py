import unittest

from fastmlx.schedule import cosine_decay


class TestSchedule(unittest.TestCase):
    """Tests for the cosine decay scheduler."""

    def test_cosine_decay(self) -> None:
        lr = cosine_decay(step=0, cycle_length=10, init_lr=1.0)
        self.assertAlmostEqual(lr, 1.0)
        lr_mid = cosine_decay(step=5, cycle_length=10, init_lr=1.0)
        self.assertLess(lr_mid, 1.0)


if __name__ == "__main__":
    unittest.main()
