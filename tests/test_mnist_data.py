import unittest
from unittest.mock import patch
import numpy as np
import mlx.core as mx
from fastmlx.dataset.data import mnist


class TestMNISTLoadData(unittest.TestCase):
    def test_load_data_converts_numpy_scalars(self):
        def fake_load_mnist(train=True):
            # return two samples with numpy arrays and numpy scalar labels
            return [
                {"image": np.zeros((28, 28, 1), dtype=np.uint8), "label": np.int64(1)},
                {"image": np.ones((28, 28, 1), dtype=np.uint8), "label": np.int64(2)},
            ]

        with patch("fastmlx.dataset.data.mnist.mlx_mnist.load_mnist", side_effect=fake_load_mnist):
            train, test = mnist.load_data()
            # ensure arrays are MX arrays
            self.assertIsInstance(train.data["x"], mx.array)
            self.assertIsInstance(train.data["y"], mx.array)
            self.assertEqual(train.data["x"].shape, (2, 28, 28, 1))
            self.assertTrue(mx.array_equal(train.data["y"], mx.array([1, 2])))


if __name__ == "__main__":
    unittest.main()
