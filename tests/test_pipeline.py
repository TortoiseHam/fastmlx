"""Tests for the Pipeline data processing."""

import unittest

import mlx.core as mx
import numpy as np

import fastmlx as fe
from fastmlx.dataset import MLXDataset
from fastmlx.op import Batch, DynamicBatch, FilteredData, Minmax, Op


class DropEvenIndices(Op):
    """Test op that filters out samples at even indices."""

    def __init__(self):
        super().__init__(inputs=["idx"], outputs=[])
        self.call_count = 0

    def forward(self, data, state):
        idx = int(data)
        self.call_count += 1
        if idx % 2 == 0:
            return FilteredData()
        return data


class AddOne(Op):
    """Simple op that adds 1 to input."""

    def __init__(self, inputs="x", outputs="x"):
        super().__init__(inputs=inputs, outputs=outputs)

    def forward(self, data, state):
        return data + 1


class MultiplyByTwo(Op):
    """Batch op that multiplies by 2."""

    def __init__(self, inputs="x", outputs="x"):
        super().__init__(inputs=inputs, outputs=outputs)

    def forward(self, data, state):
        return data * 2


class RecordMode(Op):
    """Records the mode it was called with."""

    def __init__(self, inputs="x", outputs="x"):
        super().__init__(inputs=inputs, outputs=outputs)
        self.modes = []

    def forward(self, data, state):
        self.modes.append(state.get("mode"))
        return data


class TestPipeline(unittest.TestCase):
    """Tests for the :class:`Pipeline` data processing."""

    def test_pipeline_ops(self) -> None:
        """Ensure ops run and produce expected results."""
        data = MLXDataset({"x": mx.zeros((4, 28, 28, 1), dtype=mx.uint8)})
        pipe = fe.Pipeline(
            train_data=data,
            batch_size=2,
            ops=[Minmax("x", "x")],
            num_process=0,
        )
        loader = pipe.get_loader("train")
        batch = next(iter(loader))
        self.assertEqual(batch["x"].shape, (2, 28, 28, 1))
        self.assertTrue(mx.all(batch["x"] == 0.0).item())

    def test_pipeline_threaded(self) -> None:
        """Ensure threaded loading behaves like single-threaded."""
        data = MLXDataset({"x": mx.zeros((4, 28, 28, 1), dtype=mx.uint8)})
        pipe = fe.Pipeline(
            train_data=data,
            batch_size=2,
            ops=[Minmax("x", "x")],
            num_process=2,
        )
        loader = pipe.get_loader("train")
        batch = next(iter(loader))
        self.assertEqual(batch["x"].shape, (2, 28, 28, 1))
        self.assertTrue(mx.all(batch["x"] == 0.0).item())


class TestFilteredData(unittest.TestCase):
    """Tests for FilteredData functionality."""

    def test_filtered_data_sentinel(self) -> None:
        """FilteredData is falsy and has correct attributes."""
        fd = FilteredData()
        self.assertFalse(fd)
        self.assertTrue(fd.replacement)

        fd_no_replace = FilteredData(replacement=False)
        self.assertFalse(fd_no_replace.replacement)

    def test_filtering_drops_samples(self) -> None:
        """Samples returning FilteredData are excluded from batches."""
        # Create dataset with indices 0-9
        data = [{"x": np.array([i]), "idx": np.array([i])} for i in range(10)]

        filter_op = DropEvenIndices()
        pipe = fe.Pipeline(
            train_data=data,
            batch_size=3,
            ops=[filter_op],
            num_process=0,
        )

        # Collect all batches
        all_indices = []
        for batch in pipe.get_loader("train", shuffle=False):
            indices = batch["idx"].tolist()
            all_indices.extend([idx[0] for idx in indices])

        # Should only have odd indices: 1, 3, 5, 7, 9
        self.assertEqual(sorted(all_indices), [1, 3, 5, 7, 9])

    def test_filtering_affects_batch_size(self) -> None:
        """When samples are filtered, batch sizes may vary."""
        # 10 samples, filter half, batch_size=3
        data = [{"x": np.array([i]), "idx": np.array([i])} for i in range(10)]

        pipe = fe.Pipeline(
            train_data=data,
            batch_size=3,
            ops=[DropEvenIndices()],
            num_process=0,
        )

        batches = list(pipe.get_loader("train", shuffle=False))
        # 5 samples remaining: batches of [3, 2]
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0]["x"].shape[0], 3)
        self.assertEqual(batches[1]["x"].shape[0], 2)


class TestBatchOp(unittest.TestCase):
    """Tests for Batch op functionality."""

    def test_explicit_batch_size(self) -> None:
        """Batch op overrides pipeline batch_size."""
        data = [{"x": np.array([i])} for i in range(10)]

        pipe = fe.Pipeline(
            train_data=data,
            batch_size=2,  # This should be overridden
            ops=[Batch(batch_size=4)],
            num_process=0,
        )

        batch = next(iter(pipe.get_loader("train", shuffle=False)))
        self.assertEqual(batch["x"].shape[0], 4)

    def test_drop_last(self) -> None:
        """drop_last=True drops incomplete final batch."""
        data = [{"x": np.array([i])} for i in range(10)]

        pipe = fe.Pipeline(
            train_data=data,
            ops=[Batch(batch_size=4, drop_last=True)],
            num_process=0,
        )

        batches = list(pipe.get_loader("train", shuffle=False))
        # 10 samples / 4 = 2 full batches, drop last 2
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertEqual(batch["x"].shape[0], 4)

    def test_padding_variable_length(self) -> None:
        """Batch pads variable-length sequences."""
        data = [
            {"tokens": np.array([1, 2, 3])},
            {"tokens": np.array([4, 5])},
            {"tokens": np.array([6, 7, 8, 9])},
        ]

        pipe = fe.Pipeline(
            train_data=data,
            ops=[Batch(batch_size=3, pad_value=0)],
            num_process=0,
        )

        batch = next(iter(pipe.get_loader("train", shuffle=False)))
        # Should be padded to max length (4)
        self.assertEqual(batch["tokens"].shape, (3, 4))
        # Check padding values
        tokens = np.array(batch["tokens"])
        self.assertEqual(tokens[0, 3], 0)  # First seq padded
        self.assertEqual(tokens[1, 2], 0)  # Second seq padded
        self.assertEqual(tokens[1, 3], 0)

    def test_padding_error_without_pad_value(self) -> None:
        """Variable-length without pad_value raises error."""
        data = [
            {"tokens": np.array([1, 2, 3])},
            {"tokens": np.array([4, 5])},
        ]

        pipe = fe.Pipeline(
            train_data=data,
            ops=[Batch(batch_size=2)],  # No pad_value
            num_process=0,
        )

        with self.assertRaises(Exception) as ctx:
            list(pipe.get_loader("train", shuffle=False))
        self.assertIn("padding", str(ctx.exception).lower())

    def test_custom_collate_fn(self) -> None:
        """Custom collate function is used."""
        data = [{"x": np.array([i]), "y": np.array([i * 2])} for i in range(4)]

        def custom_collate(samples):
            # Only return x, ignore y, and square it
            x_vals = np.stack([s["x"] for s in samples])
            return {"x_squared": mx.array(x_vals ** 2)}

        pipe = fe.Pipeline(
            train_data=data,
            ops=[Batch(batch_size=2, collate_fn=custom_collate)],
            num_process=0,
        )

        batch = next(iter(pipe.get_loader("train", shuffle=False)))
        self.assertIn("x_squared", batch)
        self.assertNotIn("y", batch)
        self.assertEqual(batch["x_squared"].tolist(), [[0], [1]])


class TestPrePostBatchOps(unittest.TestCase):
    """Tests for sample ops vs batch ops separation."""

    def test_ops_split_by_batch(self) -> None:
        """Ops before Batch are sample ops, after are batch ops."""
        data = [{"x": np.array([float(i)])} for i in range(4)]

        # AddOne runs on samples (numpy), MultiplyByTwo on batches (MLX)
        pipe = fe.Pipeline(
            train_data=data,
            ops=[
                AddOne(inputs="x", outputs="x"),  # Sample op: x + 1
                Batch(batch_size=2),
                MultiplyByTwo(inputs="x", outputs="x"),  # Batch op: x * 2
            ],
            num_process=0,
        )

        batch = next(iter(pipe.get_loader("train", shuffle=False)))
        # Original [0, 1] -> +1 = [1, 2] -> *2 = [2, 4]
        expected = [[2.0], [4.0]]
        self.assertEqual(batch["x"].tolist(), expected)

    def test_multiple_batch_ops_error(self) -> None:
        """Multiple Batch ops raise an error."""
        data = [{"x": np.array([i])} for i in range(4)]

        with self.assertRaises(ValueError) as ctx:
            fe.Pipeline(
                train_data=data,
                ops=[
                    Batch(batch_size=2),
                    Batch(batch_size=4),  # Second Batch - error
                ],
            )
        self.assertIn("Only one Batch op", str(ctx.exception))

    def test_default_batch_when_no_batch_op(self) -> None:
        """Pipeline creates default Batch when none specified."""
        data = [{"x": np.array([i])} for i in range(4)]

        pipe = fe.Pipeline(
            train_data=data,
            batch_size=2,
            ops=[AddOne(inputs="x", outputs="x")],
            num_process=0,
        )

        # Should still work with default batching
        self.assertEqual(pipe.batch_size, 2)
        batch = next(iter(pipe.get_loader("train", shuffle=False)))
        self.assertEqual(batch["x"].shape[0], 2)


class TestDynamicBatch(unittest.TestCase):
    """Tests for DynamicBatch functionality."""

    def test_dynamic_batch_groups_by_tokens(self) -> None:
        """DynamicBatch groups samples to maximize token usage."""
        # Variable length sequences
        data = [
            {"tokens": np.arange(10)},   # 10 tokens
            {"tokens": np.arange(20)},   # 20 tokens
            {"tokens": np.arange(5)},    # 5 tokens
            {"tokens": np.arange(15)},   # 15 tokens
            {"tokens": np.arange(8)},    # 8 tokens
        ]

        pipe = fe.Pipeline(
            train_data=data,
            ops=[
                DynamicBatch(
                    max_tokens=50,  # Max 50 tokens per batch
                    max_batch_size=4,
                    size_fn=lambda s: len(s["tokens"]),
                    pad_value=0,
                )
            ],
            num_process=0,
        )

        batches = list(pipe.get_loader("train", shuffle=False))

        # Verify no batch exceeds token limit (approximately)
        for batch in batches:
            batch_size = batch["tokens"].shape[0]
            max_len = batch["tokens"].shape[1]
            # With padding, tokens = batch_size * max_len
            self.assertLessEqual(batch_size * max_len, 50 * 1.5)  # Allow some slack


class TestPipelineModes(unittest.TestCase):
    """Tests for mode handling in pipeline."""

    def test_mode_passed_to_ops(self) -> None:
        """Ops receive correct mode in state."""
        data = [{"x": np.array([i])} for i in range(4)]
        eval_data = [{"x": np.array([i])} for i in range(2)]

        record_op = RecordMode()
        pipe = fe.Pipeline(
            train_data=data,
            eval_data=eval_data,
            batch_size=2,
            ops=[record_op],
            num_process=0,
        )

        # Consume train loader
        list(pipe.get_loader("train", shuffle=False))
        train_modes = record_op.modes.copy()
        record_op.modes.clear()

        # Consume eval loader
        list(pipe.get_loader("eval", shuffle=False))
        eval_modes = record_op.modes

        self.assertTrue(all(m == "train" for m in train_modes))
        self.assertTrue(all(m == "eval" for m in eval_modes))

    def test_shuffle_default_by_mode(self) -> None:
        """Train shuffles by default, eval doesn't."""
        data = [{"x": np.array([i])} for i in range(100)]

        pipe = fe.Pipeline(
            train_data=data,
            eval_data=data,
            batch_size=100,
            num_process=0,
        )

        # Eval should be deterministic (no shuffle by default)
        eval_batch1 = next(iter(pipe.get_loader("eval")))
        eval_batch2 = next(iter(pipe.get_loader("eval")))
        self.assertTrue(mx.array_equal(eval_batch1["x"], eval_batch2["x"]))


class TestPipelineIteration(unittest.TestCase):
    """Tests for pipeline iteration behavior."""

    def test_pipeline_iter(self) -> None:
        """Pipeline is directly iterable for training."""
        data = [{"x": np.array([i])} for i in range(4)]

        pipe = fe.Pipeline(train_data=data, batch_size=2, num_process=0)

        batches = list(pipe)
        self.assertEqual(len(batches), 2)

    def test_empty_dataset(self) -> None:
        """Empty dataset yields no batches."""
        pipe = fe.Pipeline(train_data=[], batch_size=2, num_process=0)

        batches = list(pipe.get_loader("train"))
        self.assertEqual(len(batches), 0)

    def test_batch_size_property(self) -> None:
        """batch_size property returns correct value."""
        pipe = fe.Pipeline(
            train_data=[{"x": np.array([1])}],
            ops=[Batch(batch_size=16)],
        )
        self.assertEqual(pipe.batch_size, 16)


if __name__ == "__main__":
    unittest.main()
