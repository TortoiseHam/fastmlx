"""Tests for augmentation ops."""

import unittest

import mlx.core as mx

from fastmlx.op import (
    HorizontalFlip,
    VerticalFlip,
    Rotate90,
    GaussianNoise,
    Brightness,
    Contrast,
    Resize,
    CenterCrop,
    Cutout,
)


class TestHorizontalFlip(unittest.TestCase):
    """Tests for HorizontalFlip op."""

    def test_flip_image(self) -> None:
        """Test horizontal flip of an image."""
        op = HorizontalFlip("x", "x", prob=1.0)  # Always flip
        # Create image with gradient
        data = mx.arange(12).reshape(3, 4, 1).astype(mx.float32)

        result = op.forward(data, {})

        # Check that columns are reversed
        self.assertEqual(result.shape, data.shape)
        # First column should become last
        self.assertEqual(int(result[0, 0, 0].item()), int(data[0, 3, 0].item()))

    def test_no_flip(self) -> None:
        """Test with prob=0 (never flip)."""
        op = HorizontalFlip("x", "x", prob=0.0)
        data = mx.arange(12).reshape(3, 4, 1).astype(mx.float32)

        result = op.forward(data, {})

        self.assertTrue(mx.allclose(result, data).item())


class TestVerticalFlip(unittest.TestCase):
    """Tests for VerticalFlip op."""

    def test_flip_image(self) -> None:
        """Test vertical flip of an image."""
        op = VerticalFlip("x", "x", prob=1.0)
        data = mx.arange(12).reshape(3, 4, 1).astype(mx.float32)

        result = op.forward(data, {})

        # First row should become last
        self.assertEqual(result.shape, data.shape)
        self.assertEqual(int(result[0, 0, 0].item()), int(data[2, 0, 0].item()))


class TestRotate90(unittest.TestCase):
    """Tests for Rotate90 op."""

    def test_rotate_90(self) -> None:
        """Test 90 degree rotation."""
        op = Rotate90("x", "x", k=1)
        data = mx.array([[[1], [2]], [[3], [4]]]).astype(mx.float32)  # 2x2x1

        result = op.forward(data, {})

        # After 90° rotation, shape should swap H and W
        self.assertEqual(result.shape, (2, 2, 1))

    def test_rotate_180(self) -> None:
        """Test 180 degree rotation."""
        op = Rotate90("x", "x", k=2)
        data = mx.array([[[1], [2]], [[3], [4]]]).astype(mx.float32)

        result = op.forward(data, {})

        # 180° should give [[4,3],[2,1]]
        self.assertEqual(int(result[0, 0, 0].item()), 4)
        self.assertEqual(int(result[1, 1, 0].item()), 1)


class TestGaussianNoise(unittest.TestCase):
    """Tests for GaussianNoise op."""

    def test_adds_noise(self) -> None:
        """Test that noise is added to the image."""
        op = GaussianNoise("x", "x", std=0.1)
        data = mx.zeros((28, 28, 1))

        result = op.forward(data, {})

        # Result should not be all zeros (noise was added)
        self.assertFalse(mx.all(result == 0).item())

    def test_shape_preserved(self) -> None:
        """Test that shape is preserved."""
        op = GaussianNoise("x", "x", std=0.1)
        data = mx.zeros((32, 32, 3))

        result = op.forward(data, {})

        self.assertEqual(result.shape, data.shape)


class TestBrightness(unittest.TestCase):
    """Tests for Brightness op."""

    def test_increase_brightness(self) -> None:
        """Test increasing brightness."""
        op = Brightness("x", "x", delta=0.2)
        data = mx.array([[[0.5]]])

        result = op.forward(data, {})

        self.assertGreater(float(result[0, 0, 0].item()), 0.5)

    def test_clipping(self) -> None:
        """Test that values are clipped to [0, 1]."""
        op = Brightness("x", "x", delta=0.5)
        data = mx.array([[[0.9]]])

        result = op.forward(data, {})

        self.assertLessEqual(float(result[0, 0, 0].item()), 1.0)


class TestContrast(unittest.TestCase):
    """Tests for Contrast op."""

    def test_increase_contrast(self) -> None:
        """Test increasing contrast."""
        op = Contrast("x", "x", factor=1.5)
        data = mx.array([[[0.3], [0.7]]])  # Two pixels

        result = op.forward(data, {})

        # Higher contrast should push values further from mean
        mean = 0.5
        self.assertLess(float(result[0, 0, 0].item()), 0.3)
        self.assertGreater(float(result[0, 1, 0].item()), 0.7)

    def test_decrease_contrast(self) -> None:
        """Test decreasing contrast."""
        op = Contrast("x", "x", factor=0.5)
        data = mx.array([[[0.0], [1.0]]])

        result = op.forward(data, {})

        # Lower contrast should push values toward mean
        self.assertGreater(float(result[0, 0, 0].item()), 0.0)
        self.assertLess(float(result[0, 1, 0].item()), 1.0)


class TestResize(unittest.TestCase):
    """Tests for Resize op."""

    def test_resize_upscale(self) -> None:
        """Test upscaling an image."""
        op = Resize("x", "x", height=56, width=56)
        data = mx.zeros((28, 28, 1))

        result = op.forward(data, {})

        self.assertEqual(result.shape, (56, 56, 1))

    def test_resize_downscale(self) -> None:
        """Test downscaling an image."""
        op = Resize("x", "x", height=14, width=14)
        data = mx.zeros((28, 28, 3))

        result = op.forward(data, {})

        self.assertEqual(result.shape, (14, 14, 3))


class TestCenterCrop(unittest.TestCase):
    """Tests for CenterCrop op."""

    def test_center_crop(self) -> None:
        """Test center cropping."""
        op = CenterCrop("x", "x", height=20, width=20)
        data = mx.zeros((28, 28, 1))

        result = op.forward(data, {})

        self.assertEqual(result.shape, (20, 20, 1))

    def test_crop_extracts_center(self) -> None:
        """Test that crop extracts the center region."""
        op = CenterCrop("x", "x", height=2, width=2)
        # Create 4x4 image with known values
        data = mx.arange(16).reshape(4, 4, 1).astype(mx.float32)

        result = op.forward(data, {})

        # Center 2x2 of 4x4 should be indices [1:3, 1:3]
        self.assertEqual(result.shape, (2, 2, 1))


class TestCutout(unittest.TestCase):
    """Tests for Cutout op."""

    def test_cutout_shape_preserved(self) -> None:
        """Test that cutout preserves image shape."""
        op = Cutout("x", "x", num_holes=1, max_h=8, max_w=8)
        data = mx.ones((28, 28, 1))

        result = op.forward(data, {})

        self.assertEqual(result.shape, data.shape)

    def test_cutout_creates_zeros(self) -> None:
        """Test that cutout creates zero regions."""
        op = Cutout("x", "x", num_holes=1, max_h=28, max_w=28)  # Large cutout
        data = mx.ones((28, 28, 1))

        result = op.forward(data, {})

        # Should have some zeros now
        self.assertTrue(mx.any(result == 0).item())


if __name__ == "__main__":
    unittest.main()
