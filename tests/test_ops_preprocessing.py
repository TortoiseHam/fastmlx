"""Tests for preprocessing ops."""

import unittest

import mlx.core as mx

from fastmlx.op import (
    Minmax,
    Normalize,
    ExpandDims,
    Onehot,
    Squeeze,
    Reshape,
    Cast,
    Clip,
    LambdaOp,
)


class TestMinmax(unittest.TestCase):
    """Tests for Minmax op."""

    def test_scale_uint8(self) -> None:
        """Test scaling uint8 data to [0, 1]."""
        op = Minmax("x", "x")
        data = mx.array([0, 127, 255], dtype=mx.uint8)

        result = op.forward(data, {})

        self.assertAlmostEqual(float(result[0].item()), 0.0, places=5)
        self.assertAlmostEqual(float(result[1].item()), 127 / 255, places=5)
        self.assertAlmostEqual(float(result[2].item()), 1.0, places=5)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = Minmax("x", "x")
        data = mx.array([0, 255], dtype=mx.uint8)

        result = op.forward(data, {})

        self.assertEqual(result.dtype, mx.float32)

    def test_batch_images(self) -> None:
        """Test with batch of images."""
        op = Minmax("x", "x")
        data = mx.zeros((4, 28, 28, 1), dtype=mx.uint8)

        result = op.forward(data, {})

        self.assertEqual(result.shape, (4, 28, 28, 1))
        self.assertTrue(mx.all(result == 0.0).item())


class TestNormalize(unittest.TestCase):
    """Tests for Normalize op."""

    def test_normalize_mean_std(self) -> None:
        """Test normalization with mean and std."""
        op = Normalize("x", "x", mean=0.5, std=0.5)
        data = mx.array([0.0, 0.5, 1.0])

        result = op.forward(data, {})

        # (0 - 0.5) / 0.5 = -1, (0.5 - 0.5) / 0.5 = 0, (1 - 0.5) / 0.5 = 1
        self.assertAlmostEqual(float(result[0].item()), -1.0, places=5)
        self.assertAlmostEqual(float(result[1].item()), 0.0, places=5)
        self.assertAlmostEqual(float(result[2].item()), 1.0, places=5)


class TestExpandDims(unittest.TestCase):
    """Tests for ExpandDims op."""

    def test_expand_last_axis(self) -> None:
        """Test expanding last axis."""
        op = ExpandDims("x", "x", axis=-1)
        data = mx.zeros((28, 28))

        result = op.forward(data, {})

        self.assertEqual(result.shape, (28, 28, 1))

    def test_expand_first_axis(self) -> None:
        """Test expanding first axis."""
        op = ExpandDims("x", "x", axis=0)
        data = mx.zeros((28, 28))

        result = op.forward(data, {})

        self.assertEqual(result.shape, (1, 28, 28))


class TestOnehot(unittest.TestCase):
    """Tests for Onehot op."""

    def test_onehot_encoding(self) -> None:
        """Test one-hot encoding."""
        op = Onehot("y", "y", num_classes=10)
        data = mx.array([0, 3, 9])

        result = op.forward(data, {})

        self.assertEqual(result.shape, (3, 10))
        # Check class 0
        self.assertEqual(float(result[0, 0].item()), 1.0)
        self.assertEqual(float(result[0, 1].item()), 0.0)
        # Check class 3
        self.assertEqual(float(result[1, 3].item()), 1.0)
        # Check class 9
        self.assertEqual(float(result[2, 9].item()), 1.0)

    def test_single_label(self) -> None:
        """Test with single label."""
        op = Onehot("y", "y", num_classes=5)
        data = mx.array(2)

        result = op.forward(data, {})

        expected = mx.array([0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertTrue(mx.allclose(result, expected).item())


class TestSqueeze(unittest.TestCase):
    """Tests for Squeeze op."""

    def test_squeeze_all(self) -> None:
        """Test squeezing all singleton dimensions."""
        op = Squeeze("x", "x")
        data = mx.zeros((1, 28, 28, 1))

        result = op.forward(data, {})

        self.assertEqual(result.shape, (28, 28))

    def test_squeeze_specific_axis(self) -> None:
        """Test squeezing specific axis."""
        op = Squeeze("x", "x", axis=0)
        data = mx.zeros((1, 28, 28, 1))

        result = op.forward(data, {})

        self.assertEqual(result.shape, (28, 28, 1))


class TestReshape(unittest.TestCase):
    """Tests for Reshape op."""

    def test_reshape_flatten(self) -> None:
        """Test flattening with reshape."""
        op = Reshape("x", "x", shape=(-1,))
        data = mx.zeros((28, 28))

        result = op.forward(data, {})

        self.assertEqual(result.shape, (784,))

    def test_reshape_explicit(self) -> None:
        """Test explicit reshape."""
        op = Reshape("x", "x", shape=(4, 196))
        data = mx.zeros((28, 28))

        result = op.forward(data, {})

        self.assertEqual(result.shape, (4, 196))


class TestCast(unittest.TestCase):
    """Tests for Cast op."""

    def test_cast_to_float16(self) -> None:
        """Test casting to float16."""
        op = Cast("x", "x", dtype=mx.float16)
        data = mx.array([1.0, 2.0, 3.0])

        result = op.forward(data, {})

        self.assertEqual(result.dtype, mx.float16)

    def test_cast_to_int32(self) -> None:
        """Test casting to int32."""
        op = Cast("x", "x", dtype=mx.int32)
        data = mx.array([1.5, 2.7, 3.1])

        result = op.forward(data, {})

        self.assertEqual(result.dtype, mx.int32)
        self.assertEqual(int(result[0].item()), 1)


class TestClip(unittest.TestCase):
    """Tests for Clip op."""

    def test_clip_values(self) -> None:
        """Test clipping values."""
        op = Clip("x", "x", min_val=0.0, max_val=1.0)
        data = mx.array([-0.5, 0.5, 1.5])

        result = op.forward(data, {})

        self.assertAlmostEqual(float(result[0].item()), 0.0, places=5)
        self.assertAlmostEqual(float(result[1].item()), 0.5, places=5)
        self.assertAlmostEqual(float(result[2].item()), 1.0, places=5)


class TestLambdaOp(unittest.TestCase):
    """Tests for LambdaOp."""

    def test_lambda_function(self) -> None:
        """Test applying lambda function."""
        op = LambdaOp("x", "x", fn=lambda x: x ** 2)
        data = mx.array([2.0, 3.0, 4.0])

        result = op.forward(data, {})

        expected = mx.array([4.0, 9.0, 16.0])
        self.assertTrue(mx.allclose(result, expected).item())

    def test_lambda_with_state(self) -> None:
        """Test lambda that uses state."""
        op = LambdaOp("x", "x", fn=lambda x, s: x * s.get("multiplier", 1))
        data = mx.array([1.0, 2.0])
        state = {"multiplier": 3}

        result = op.forward(data, state)

        expected = mx.array([3.0, 6.0])
        self.assertTrue(mx.allclose(result, expected).item())


if __name__ == "__main__":
    unittest.main()
