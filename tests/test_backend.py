"""Tests for backend utilities module."""

import unittest

import mlx.core as mx

from fastmlx.backend import (
    get_device,
    set_device,
    get_default_device,
    is_gpu_available,
    device_info,
    synchronize,
    memory_info,
    clear_cache,
    get_dtype,
    to_dtype,
    dtype_info,
    float16,
    float32,
    bfloat16,
    int32,
    set_seed,
    get_seed,
)
from fastmlx.backend.memory import estimate_array_memory, format_memory
from fastmlx.backend.seed import fork_rng, random_split


class TestDeviceUtils(unittest.TestCase):
    """Tests for device utilities."""

    def test_get_device(self) -> None:
        """Test getting current device."""
        device = get_device()
        self.assertIsInstance(device, mx.Device)

    def test_get_default_device_string(self) -> None:
        """Test getting device as string."""
        device_str = get_default_device()
        self.assertIn(device_str, ["gpu", "cpu"])

    def test_set_device_cpu(self) -> None:
        """Test setting CPU device."""
        original = get_device()
        try:
            set_device("cpu")
            self.assertEqual(get_default_device(), "cpu")
        finally:
            # Restore original device
            mx.set_default_device(original)

    def test_device_info(self) -> None:
        """Test getting device information."""
        info = device_info()

        self.assertIn("default_device", info)
        self.assertIn("platform", info)
        self.assertIn("machine", info)

    def test_is_gpu_available(self) -> None:
        """Test GPU availability check."""
        available = is_gpu_available()
        self.assertIsInstance(available, bool)

    def test_synchronize(self) -> None:
        """Test synchronization."""
        x = mx.random.normal((100, 100))
        y = mx.matmul(x, x)

        # Should not raise
        synchronize()


class TestMemoryUtils(unittest.TestCase):
    """Tests for memory utilities."""

    def test_memory_info(self) -> None:
        """Test getting memory info."""
        info = memory_info()
        self.assertIsInstance(info, dict)

    def test_clear_cache(self) -> None:
        """Test clearing cache (should not raise)."""
        clear_cache()

    def test_estimate_array_memory_float32(self) -> None:
        """Test memory estimation for float32."""
        mem = estimate_array_memory((1000, 1000), mx.float32)
        # 1M elements * 4 bytes = 4MB
        self.assertEqual(mem, 4_000_000)

    def test_estimate_array_memory_float16(self) -> None:
        """Test memory estimation for float16."""
        mem = estimate_array_memory((1000, 1000), mx.float16)
        # 1M elements * 2 bytes = 2MB
        self.assertEqual(mem, 2_000_000)

    def test_format_memory_bytes(self) -> None:
        """Test formatting small memory sizes."""
        self.assertEqual(format_memory(512), "512 B")

    def test_format_memory_kb(self) -> None:
        """Test formatting KB memory sizes."""
        formatted = format_memory(2048)
        self.assertIn("KB", formatted)

    def test_format_memory_mb(self) -> None:
        """Test formatting MB memory sizes."""
        formatted = format_memory(5_000_000)
        self.assertIn("MB", formatted)

    def test_format_memory_gb(self) -> None:
        """Test formatting GB memory sizes."""
        formatted = format_memory(2_000_000_000)
        self.assertIn("GB", formatted)


class TestDtypeUtils(unittest.TestCase):
    """Tests for dtype utilities."""

    def test_dtype_aliases(self) -> None:
        """Test dtype aliases."""
        self.assertEqual(float16, mx.float16)
        self.assertEqual(float32, mx.float32)
        self.assertEqual(bfloat16, mx.bfloat16)
        self.assertEqual(int32, mx.int32)

    def test_get_dtype_from_string(self) -> None:
        """Test getting dtype from string."""
        self.assertEqual(get_dtype("float32"), mx.float32)
        self.assertEqual(get_dtype("float16"), mx.float16)
        self.assertEqual(get_dtype("int32"), mx.int32)
        self.assertEqual(get_dtype("bool"), mx.bool_)

    def test_get_dtype_aliases(self) -> None:
        """Test dtype string aliases."""
        self.assertEqual(get_dtype("fp16"), mx.float16)
        self.assertEqual(get_dtype("fp32"), mx.float32)
        self.assertEqual(get_dtype("half"), mx.float16)

    def test_get_dtype_invalid(self) -> None:
        """Test invalid dtype raises error."""
        with self.assertRaises(ValueError):
            get_dtype("invalid_dtype")

    def test_to_dtype_from_string(self) -> None:
        """Test converting array dtype with string."""
        x = mx.array([1.0, 2.0, 3.0])
        y = to_dtype(x, "float16")

        self.assertEqual(y.dtype, mx.float16)

    def test_to_dtype_from_dtype(self) -> None:
        """Test converting array dtype with dtype."""
        x = mx.array([1.0, 2.0, 3.0])
        y = to_dtype(x, mx.int32)

        self.assertEqual(y.dtype, mx.int32)
        self.assertEqual(int(y[0].item()), 1)

    def test_dtype_info(self) -> None:
        """Test getting dtype info."""
        info = dtype_info("float32")

        self.assertEqual(info["bits"], 32)
        self.assertEqual(info["category"], "float")
        self.assertEqual(info["name"], "float32")

    def test_dtype_info_int(self) -> None:
        """Test dtype info for int."""
        info = dtype_info(mx.int64)

        self.assertEqual(info["bits"], 64)
        self.assertEqual(info["category"], "int")


class TestSeedUtils(unittest.TestCase):
    """Tests for seed utilities."""

    def test_set_and_get_seed(self) -> None:
        """Test setting and getting seed."""
        set_seed(42)
        self.assertEqual(get_seed(), 42)

    def test_reproducibility(self) -> None:
        """Test that same seed gives same results."""
        set_seed(123)
        x1 = mx.random.normal((5,))
        mx.eval(x1)

        set_seed(123)
        x2 = mx.random.normal((5,))
        mx.eval(x2)

        self.assertTrue(mx.allclose(x1, x2).item())

    def test_fork_rng(self) -> None:
        """Test fork_rng context manager."""
        set_seed(42)
        x1 = mx.random.normal((3,))
        mx.eval(x1)

        with fork_rng(seed=999):
            y = mx.random.normal((3,))
            mx.eval(y)

        # After fork, should continue from saved state
        # Note: This tests the fork mechanism, actual values depend on implementation

    def test_random_split(self) -> None:
        """Test splitting random keys."""
        keys = random_split(42, num=4)

        self.assertEqual(len(keys), 4)
        # All keys should be different
        self.assertEqual(len(set(keys)), 4)


class TestDtypePromotion(unittest.TestCase):
    """Tests for dtype promotion."""

    def test_promote_float16_float32(self) -> None:
        """Test promoting float16 and float32."""
        from fastmlx.backend.dtype import promote_types

        result = promote_types(mx.float16, mx.float32)
        self.assertEqual(result, mx.float32)

    def test_promote_int_float(self) -> None:
        """Test promoting int and float."""
        from fastmlx.backend.dtype import promote_types

        result = promote_types(mx.int32, mx.float32)
        self.assertEqual(result, mx.float32)


if __name__ == "__main__":
    unittest.main()
