import unittest

import mlx.core as mx

from fastmlx.architecture import LeNet


class TestLeNet(unittest.TestCase):
    """Unit tests for the :class:`LeNet` architecture."""

    def test_forward(self) -> None:
        """Ensure that the network produces the expected output shape."""

        model = LeNet()
        mx.eval(model.parameters())
        x = mx.random.uniform(shape=(2, 28, 28, 1))
        y = model(x)
        self.assertEqual(y.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()

