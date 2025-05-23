import unittest

import mlx.core as mx

import fastmlx as fe
from fastmlx.dataset import MLXDataset
from fastmlx.op import Minmax


class TestPipeline(unittest.TestCase):
    """Tests for the :class:`Pipeline` data processing."""

    def test_pipeline_ops(self) -> None:
        """Ensure ops run and produce expected results."""

        data = MLXDataset({"x": mx.zeros((4, 28, 28, 1), dtype=mx.uint8)})
        pipe = fe.Pipeline(
            train_data=data,
            batch_size=2,
            ops=[Minmax("x", "x")],
        )
        loader = pipe.get_loader("train")
        batch = next(iter(loader))
        self.assertEqual(batch["x"].shape, (2, 28, 28, 1))
        self.assertTrue(mx.all(batch["x"] == 0.0).item())


if __name__ == "__main__":
    unittest.main()
