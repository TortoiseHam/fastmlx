"""Tests for the Network class."""

import unittest

import mlx.core as mx

from fastmlx.network import Network
from fastmlx.op import Op


class MockOp(Op):
    """A simple op that doubles its input."""

    def __init__(self, inputs: str, outputs: str) -> None:
        super().__init__(inputs, outputs)

    def forward(self, data, state):
        return data * 2


class AddOp(Op):
    """Op that adds two inputs."""

    def __init__(self, inputs, outputs) -> None:
        super().__init__(inputs, outputs)

    def forward(self, data, state):
        return data[0] + data[1]


class TestNetwork(unittest.TestCase):
    """Tests for the Network class."""

    def test_single_op(self) -> None:
        """Test network with a single op."""
        network = Network([MockOp("x", "y")])
        batch = {"x": mx.array([1.0, 2.0, 3.0])}
        state = {}

        result = network.run(batch, state)

        self.assertIn("y", result)
        expected = mx.array([2.0, 4.0, 6.0])
        self.assertTrue(mx.allclose(result["y"], expected).item())

    def test_chained_ops(self) -> None:
        """Test network with chained ops."""
        network = Network([
            MockOp("x", "y"),
            MockOp("y", "z"),
        ])
        batch = {"x": mx.array([1.0, 2.0])}
        state = {}

        result = network.run(batch, state)

        self.assertIn("z", result)
        # x * 2 * 2 = x * 4
        expected = mx.array([4.0, 8.0])
        self.assertTrue(mx.allclose(result["z"], expected).item())

    def test_multi_input_op(self) -> None:
        """Test op with multiple inputs."""
        network = Network([AddOp(["a", "b"], "c")])
        batch = {
            "a": mx.array([1.0, 2.0]),
            "b": mx.array([3.0, 4.0]),
        }
        state = {}

        result = network.run(batch, state)

        self.assertIn("c", result)
        expected = mx.array([4.0, 6.0])
        self.assertTrue(mx.allclose(result["c"], expected).item())

    def test_get_loss_keys(self) -> None:
        """Test loss key detection."""
        network = Network([
            MockOp("x", "y_pred"),
            MockOp("y_pred", "loss"),
            MockOp("loss", "ce"),
        ])

        loss_keys = network.get_loss_keys()

        self.assertIn("loss", loss_keys)
        self.assertIn("ce", loss_keys)
        self.assertNotIn("y_pred", loss_keys)

    def test_in_place_modification(self) -> None:
        """Test that ops can modify batch in place via outputs."""
        network = Network([MockOp("x", "x")])  # Same input/output key
        batch = {"x": mx.array([1.0, 2.0])}
        state = {}

        result = network.run(batch, state)

        expected = mx.array([2.0, 4.0])
        self.assertTrue(mx.allclose(result["x"], expected).item())


if __name__ == "__main__":
    unittest.main()
