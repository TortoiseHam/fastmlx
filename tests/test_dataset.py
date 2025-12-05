"""Tests for dataset classes."""

import unittest
import tempfile
import os
import csv

import mlx.core as mx
import numpy as np

from fastmlx.dataset import (
    MLXDataset,
    CSVDataset,
    GeneratorDataset,
    BatchDataset,
    CombinedDataset,
    InterleaveDataset,
)


class TestMLXDataset(unittest.TestCase):
    """Tests for MLXDataset."""

    def test_creation(self) -> None:
        """Test basic dataset creation."""
        data = {
            "x": mx.zeros((10, 28, 28, 1)),
            "y": mx.arange(10),
        }
        dataset = MLXDataset(data)

        self.assertEqual(len(dataset), 10)

    def test_getitem(self) -> None:
        """Test indexing into dataset."""
        data = {
            "x": mx.arange(5),
            "y": mx.arange(5) * 2,
        }
        dataset = MLXDataset(data)

        sample = dataset[2]

        self.assertEqual(int(sample["x"].item()), 2)
        self.assertEqual(int(sample["y"].item()), 4)

    def test_iteration(self) -> None:
        """Test iterating over dataset."""
        data = {"x": mx.arange(3)}
        dataset = MLXDataset(data)

        samples = [dataset[i] for i in range(len(dataset))]

        self.assertEqual(len(samples), 3)
        for i, sample in enumerate(samples):
            self.assertEqual(int(sample["x"].item()), i)

    def test_numpy_input(self) -> None:
        """Test creation from numpy arrays."""
        data = {
            "x": np.zeros((5, 10)),
            "y": np.ones(5),
        }
        dataset = MLXDataset(data)

        self.assertEqual(len(dataset), 5)
        sample = dataset[0]
        self.assertEqual(sample["x"].shape, (10,))


class TestCSVDataset(unittest.TestCase):
    """Tests for CSVDataset."""

    def setUp(self) -> None:
        """Create a temporary CSV file."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test.csv")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["feature1", "feature2", "label"])
            writer.writerow([1.0, 2.0, 0])
            writer.writerow([3.0, 4.0, 1])
            writer.writerow([5.0, 6.0, 0])

    def tearDown(self) -> None:
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_csv(self) -> None:
        """Test loading CSV file."""
        dataset = CSVDataset(self.csv_path)

        self.assertEqual(len(dataset), 3)

    def test_csv_columns(self) -> None:
        """Test CSV column access."""
        dataset = CSVDataset(self.csv_path)
        sample = dataset[0]

        self.assertIn("feature1", sample)
        self.assertIn("feature2", sample)
        self.assertIn("label", sample)


class TestGeneratorDataset(unittest.TestCase):
    """Tests for GeneratorDataset."""

    def test_from_generator(self) -> None:
        """Test creating dataset from generator."""
        def gen():
            for i in range(5):
                yield {"x": i, "y": i * 2}

        dataset = GeneratorDataset(gen)

        # Generator datasets don't support len until iterated
        samples = list(dataset)
        self.assertEqual(len(samples), 5)
        self.assertEqual(samples[2]["x"], 2)

    def test_generator_iteration(self) -> None:
        """Test iterating over generator dataset."""
        def gen():
            for i in range(3):
                yield {"val": i}

        dataset = GeneratorDataset(gen)
        values = [s["val"] for s in dataset]

        self.assertEqual(values, [0, 1, 2])


class TestBatchDataset(unittest.TestCase):
    """Tests for BatchDataset."""

    def test_batching(self) -> None:
        """Test batching functionality."""
        data = {"x": mx.arange(10)}
        base_dataset = MLXDataset(data)
        batched = BatchDataset(base_dataset, batch_size=3)

        batches = list(batched)

        self.assertEqual(len(batches), 4)  # 10 / 3 = 3 full + 1 partial
        self.assertEqual(len(batches[0]["x"]), 3)
        self.assertEqual(len(batches[-1]["x"]), 1)  # Last batch has 1 element

    def test_drop_remainder(self) -> None:
        """Test dropping incomplete batches."""
        data = {"x": mx.arange(10)}
        base_dataset = MLXDataset(data)
        batched = BatchDataset(base_dataset, batch_size=3, drop_remainder=True)

        batches = list(batched)

        self.assertEqual(len(batches), 3)  # Only full batches


class TestCombinedDataset(unittest.TestCase):
    """Tests for CombinedDataset."""

    def test_combination(self) -> None:
        """Test combining datasets."""
        ds1 = MLXDataset({"x": mx.arange(3)})
        ds2 = MLXDataset({"x": mx.arange(3, 6)})
        combined = CombinedDataset([ds1, ds2])

        self.assertEqual(len(combined), 6)

    def test_combined_access(self) -> None:
        """Test accessing combined dataset."""
        ds1 = MLXDataset({"x": mx.array([0, 1])})
        ds2 = MLXDataset({"x": mx.array([2, 3])})
        combined = CombinedDataset([ds1, ds2])

        # Access from first dataset
        self.assertEqual(int(combined[0]["x"].item()), 0)
        self.assertEqual(int(combined[1]["x"].item()), 1)
        # Access from second dataset
        self.assertEqual(int(combined[2]["x"].item()), 2)
        self.assertEqual(int(combined[3]["x"].item()), 3)


class TestInterleaveDataset(unittest.TestCase):
    """Tests for InterleaveDataset."""

    def test_interleaving(self) -> None:
        """Test interleaving datasets."""
        ds1 = MLXDataset({"x": mx.array([0, 2, 4])})
        ds2 = MLXDataset({"x": mx.array([1, 3, 5])})
        interleaved = InterleaveDataset([ds1, ds2])

        values = [int(interleaved[i]["x"].item()) for i in range(6)]

        # Should alternate between datasets
        self.assertEqual(values, [0, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
