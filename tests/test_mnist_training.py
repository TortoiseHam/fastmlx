import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import mlx.core as mx
from fastmlx.apphub import mnist as mnist_app
from fastmlx.trace.metric import Accuracy


class TestMNISTTraining(unittest.TestCase):
    """Integration test running a minimal training loop."""

    def test_fit_runs_and_saves_model(self):
        def fake_load_mnist(train=True):
            # Small dataset for fast execution
            return [
                {"image": np.zeros((28, 28, 1), dtype=np.uint8), "label": np.int64(1)},
                {"image": np.ones((28, 28, 1), dtype=np.uint8), "label": np.int64(0)},
                {"image": np.zeros((28, 28, 1), dtype=np.uint8), "label": np.int64(1)},
                {"image": np.ones((28, 28, 1), dtype=np.uint8), "label": np.int64(0)},
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("fastmlx.dataset.data.mnist.mlx_mnist.load_mnist", side_effect=fake_load_mnist):
                est = mnist_app.get_estimator(epochs=1, batch_size=2, save_dir=tmpdir)
                # Simplify accuracy computation and bypass network execution to
                # avoid triggering heavy JIT compilation during tests.
                def simple_acc(self, batch, state):
                    y = np.array(batch[self.true_key])
                    y_pred = np.array(batch[self.pred_key])
                    pred = y_pred.argmax(axis=-1)
                    self.correct += int((pred == y).sum())
                    self.total += y.shape[0]

                def fake_run(self, batch, state):
                    batch["y_pred"] = mx.zeros((batch["y"].shape[0], 10))
                    batch["ce"] = mx.array(0.0)
                    return batch

                from types import MethodType

                with patch.object(Accuracy, "on_batch_end", simple_acc), \
                     patch.object(est.network, "run", MethodType(fake_run, est.network)):
                    est.fit()
                model_path = os.path.join(tmpdir, "model.npz")
                self.assertTrue(os.path.exists(model_path))
                weights = mx.load(model_path)
                self.assertTrue(len(weights) > 0)


if __name__ == "__main__":
    unittest.main()
