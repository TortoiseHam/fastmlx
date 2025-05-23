import unittest
from unittest.mock import patch
import numpy as np
import mlx.core as mx
from fastmlx.apphub import mnist as mnist_app


class TestMNISTEstimator(unittest.TestCase):
    def test_single_training_step(self):
        def fake_load_mnist(train=True):
            # create four samples to form two batches
            return [
                {"image": np.zeros((28, 28, 1), dtype=np.uint8), "label": np.int64(1)},
                {"image": np.ones((28, 28, 1), dtype=np.uint8), "label": np.int64(0)},
                {"image": np.zeros((28, 28, 1), dtype=np.uint8), "label": np.int64(1)},
                {"image": np.ones((28, 28, 1), dtype=np.uint8), "label": np.int64(0)},
            ]

        with patch("fastmlx.dataset.data.mnist.mlx_mnist.load_mnist", side_effect=fake_load_mnist):
            est = mnist_app.get_estimator(epochs=1, batch_size=2, num_process=0)
            loader = est.pipeline.get_loader("train")
            batch = next(iter(loader))
            state = {"mode": "train"}
            est.network.run(batch, state)
            # ensure a loss key is generated
            self.assertIn("ce", batch)


if __name__ == "__main__":
    unittest.main()
