"""Tests for neural network architectures."""

import unittest

import mlx.core as mx
import mlx.nn as nn

from fastmlx.architecture import (
    LeNet,
    ResNet9,
    UNet,
    AttentionUNet,
    WideResNet,
    WideResNet16_8,
    VisionTransformer,
    ViT_Tiny,
    MultiHeadAttention,
    TransformerEncoder,
    TransformerDecoder,
    GPT,
)


class TestLeNet(unittest.TestCase):
    """Tests for LeNet architecture."""

    def test_forward_pass(self) -> None:
        """Test forward pass shape."""
        model = LeNet(num_classes=10)
        x = mx.zeros((2, 28, 28, 1))

        output = model(x)

        self.assertEqual(output.shape, (2, 10))

    def test_different_num_classes(self) -> None:
        """Test with different number of classes."""
        model = LeNet(num_classes=100)
        x = mx.zeros((1, 28, 28, 1))

        output = model(x)

        self.assertEqual(output.shape, (1, 100))


class TestResNet9(unittest.TestCase):
    """Tests for ResNet9 architecture."""

    def test_forward_pass(self) -> None:
        """Test forward pass shape."""
        model = ResNet9(num_classes=10)
        x = mx.zeros((2, 32, 32, 3))

        output = model(x)

        self.assertEqual(output.shape, (2, 10))

    def test_cifar10_input(self) -> None:
        """Test with CIFAR-10 style input."""
        model = ResNet9(num_classes=10)
        x = mx.random.normal((4, 32, 32, 3))

        output = model(x)

        self.assertEqual(output.shape, (4, 10))
        # Output should be valid logits
        self.assertFalse(mx.any(mx.isnan(output)).item())


class TestUNet(unittest.TestCase):
    """Tests for UNet architecture."""

    def test_forward_pass(self) -> None:
        """Test forward pass shape."""
        model = UNet(in_channels=1, out_channels=1)
        x = mx.zeros((1, 128, 128, 1))

        output = model(x)

        # UNet should preserve spatial dimensions
        self.assertEqual(output.shape, (1, 128, 128, 1))

    def test_multi_channel(self) -> None:
        """Test with multiple input/output channels."""
        model = UNet(in_channels=3, out_channels=2)
        x = mx.zeros((2, 64, 64, 3))

        output = model(x)

        self.assertEqual(output.shape, (2, 64, 64, 2))


class TestAttentionUNet(unittest.TestCase):
    """Tests for AttentionUNet architecture."""

    def test_forward_pass(self) -> None:
        """Test forward pass shape."""
        model = AttentionUNet(in_channels=1, out_channels=1)
        x = mx.zeros((1, 128, 128, 1))

        output = model(x)

        self.assertEqual(output.shape, (1, 128, 128, 1))


class TestWideResNet(unittest.TestCase):
    """Tests for WideResNet architectures."""

    def test_wideresnet16_8(self) -> None:
        """Test WideResNet16-8."""
        model = WideResNet16_8(num_classes=10)
        x = mx.zeros((2, 32, 32, 3))

        output = model(x)

        self.assertEqual(output.shape, (2, 10))

    def test_custom_wideresnet(self) -> None:
        """Test custom WideResNet configuration."""
        model = WideResNet(depth=28, widen_factor=10, num_classes=100)
        x = mx.zeros((1, 32, 32, 3))

        output = model(x)

        self.assertEqual(output.shape, (1, 100))


class TestMultiHeadAttention(unittest.TestCase):
    """Tests for MultiHeadAttention."""

    def test_attention_shape(self) -> None:
        """Test attention output shape."""
        attn = MultiHeadAttention(dims=64, num_heads=4)
        x = mx.zeros((2, 10, 64))

        output, _ = attn(x)

        self.assertEqual(output.shape, (2, 10, 64))

    def test_different_dims(self) -> None:
        """Test with different dimensions."""
        attn = MultiHeadAttention(dims=128, num_heads=8)
        x = mx.zeros((1, 20, 128))

        output, cache = attn(x)

        self.assertEqual(output.shape, (1, 20, 128))
        self.assertIsNotNone(cache)


class TestTransformerEncoder(unittest.TestCase):
    """Tests for TransformerEncoder."""

    def test_encoder_shape(self) -> None:
        """Test encoder output shape."""
        encoder = TransformerEncoder(dims=64, num_layers=2, num_heads=4)
        x = mx.zeros((2, 10, 64))

        output = encoder(x)

        self.assertEqual(output.shape, (2, 10, 64))

    def test_deep_encoder(self) -> None:
        """Test deeper encoder."""
        encoder = TransformerEncoder(dims=128, num_layers=6, num_heads=8)
        x = mx.random.normal((1, 50, 128))

        output = encoder(x)

        self.assertEqual(output.shape, (1, 50, 128))


class TestTransformerDecoder(unittest.TestCase):
    """Tests for TransformerDecoder."""

    def test_decoder_shape(self) -> None:
        """Test decoder output shape."""
        decoder = TransformerDecoder(dims=64, num_layers=2, num_heads=4)
        x = mx.zeros((2, 10, 64))

        output, cache = decoder(x)

        self.assertEqual(output.shape, (2, 10, 64))
        self.assertEqual(len(cache), 2)  # One cache per layer


class TestVisionTransformer(unittest.TestCase):
    """Tests for Vision Transformer."""

    def test_vit_forward(self) -> None:
        """Test ViT forward pass."""
        model = VisionTransformer(
            image_size=32,
            patch_size=8,
            num_classes=10,
            dims=64,
            depth=2,
            num_heads=4
        )
        x = mx.zeros((2, 32, 32, 3))

        output = model(x)

        self.assertEqual(output.shape, (2, 10))

    def test_vit_tiny(self) -> None:
        """Test ViT-Tiny preset."""
        model = ViT_Tiny(num_classes=10, image_size=32, patch_size=8)
        x = mx.zeros((1, 32, 32, 3))

        output = model(x)

        self.assertEqual(output.shape, (1, 10))

    def test_patch_embedding(self) -> None:
        """Test that patches are created correctly."""
        model = VisionTransformer(
            image_size=16,
            patch_size=4,
            num_classes=5,
            dims=32,
            depth=1,
            num_heads=2
        )
        x = mx.zeros((1, 16, 16, 3))

        output = model(x)

        # 16/4 = 4, so 4x4 = 16 patches
        self.assertEqual(output.shape, (1, 5))


class TestGPT(unittest.TestCase):
    """Tests for GPT model."""

    def test_gpt_forward(self) -> None:
        """Test GPT forward pass."""
        model = GPT(
            vocab_size=100,
            max_seq_len=32,
            dims=64,
            num_layers=2,
            num_heads=4
        )
        tokens = mx.array([[1, 2, 3, 4, 5]])

        logits, cache = model(tokens)

        self.assertEqual(logits.shape, (1, 5, 100))
        self.assertEqual(len(cache), 2)

    def test_gpt_with_cache(self) -> None:
        """Test GPT with KV cache for generation."""
        model = GPT(
            vocab_size=50,
            max_seq_len=16,
            dims=32,
            num_layers=1,
            num_heads=2
        )

        # Initial forward
        tokens = mx.array([[1, 2, 3]])
        logits1, cache = model(tokens)

        # Continuation with cache
        next_token = mx.array([[4]])
        logits2, new_cache = model(next_token, cache=cache)

        self.assertEqual(logits2.shape, (1, 1, 50))


class TestModelTrainability(unittest.TestCase):
    """Test that models can be trained."""

    def test_lenet_gradients(self) -> None:
        """Test that LeNet produces gradients."""
        model = LeNet(num_classes=10)
        x = mx.random.normal((2, 28, 28, 1))
        y = mx.array([0, 1])

        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y).mean()

        loss, grads = mx.value_and_grad(loss_fn)(model, x, y)

        self.assertFalse(mx.isnan(loss).item())
        # Check that some gradients exist
        self.assertTrue(len(list(grads.parameters())) > 0)

    def test_resnet9_gradients(self) -> None:
        """Test that ResNet9 produces gradients."""
        model = ResNet9(num_classes=10)
        x = mx.random.normal((2, 32, 32, 3))
        y = mx.array([0, 1])

        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y).mean()

        loss, grads = mx.value_and_grad(loss_fn)(model, x, y)

        self.assertFalse(mx.isnan(loss).item())


if __name__ == "__main__":
    unittest.main()
