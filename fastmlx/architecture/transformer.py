"""Transformer architectures for FastMLX."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.

    Example:
        >>> attn = MultiHeadAttention(dims=512, num_heads=8)
        >>> x = mx.random.normal((2, 10, 512))  # (batch, seq, dims)
        >>> out = attn(x)  # (2, 10, 512)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__()
        assert dims % num_heads == 0, "dims must be divisible by num_heads"

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dims).
            mask: Optional attention mask.
            cache: Optional KV cache for inference.

        Returns:
            Output tensor and optional updated cache.
        """
        B, L, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Handle KV cache for autoregressive generation
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)
        new_cache = (k, v)

        # Reshape for multi-head attention
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Compute attention scores
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.dims)
        out = self.out_proj(out)

        return out, new_cache


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block.

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        mlp_dims: MLP hidden dimension (default: 4 * dims).
        dropout: Dropout probability.
        norm_first: Apply normalization before attention (pre-norm).

    Example:
        >>> block = TransformerEncoderBlock(dims=512, num_heads=8)
        >>> x = mx.random.normal((2, 10, 512))
        >>> out = block(x)  # (2, 10, 512)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int = 8,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.1,
        norm_first: bool = True
    ) -> None:
        super().__init__()
        mlp_dims = mlp_dims or dims * 4

        self.norm_first = norm_first
        self.attn = MultiHeadAttention(dims, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)

        self.mlp = nn.Sequential(
            nn.Linear(dims, mlp_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dims, dims),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        if self.norm_first:
            # Pre-norm architecture
            attn_out, _ = self.attn(self.norm1(x), mask)
            x = x + self.dropout(attn_out)
            x = x + self.mlp(self.norm2(x))
        else:
            # Post-norm architecture
            attn_out, _ = self.attn(x, mask)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.mlp(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block with causal attention.

    Args:
        dims: Model dimension.
        num_heads: Number of attention heads.
        mlp_dims: MLP hidden dimension (default: 4 * dims).
        dropout: Dropout probability.
        norm_first: Apply normalization before attention (pre-norm).

    Example:
        >>> block = TransformerDecoderBlock(dims=512, num_heads=8)
        >>> x = mx.random.normal((2, 10, 512))
        >>> out = block(x)  # (2, 10, 512)
    """

    def __init__(
        self,
        dims: int,
        num_heads: int = 8,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.1,
        norm_first: bool = True
    ) -> None:
        super().__init__()
        mlp_dims = mlp_dims or dims * 4

        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(dims, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)

        self.mlp = nn.Sequential(
            nn.Linear(dims, mlp_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dims, dims),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Create causal mask if not provided
        if mask is None:
            L = x.shape[1]
            mask = mx.triu(mx.full((L, L), float("-inf")), k=1)

        if self.norm_first:
            attn_out, new_cache = self.self_attn(self.norm1(x), mask, cache)
            x = x + self.dropout(attn_out)
            x = x + self.mlp(self.norm2(x))
        else:
            attn_out, new_cache = self.self_attn(x, mask, cache)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.mlp(x))

        return x, new_cache


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks.

    Args:
        dims: Model dimension.
        num_layers: Number of encoder blocks.
        num_heads: Number of attention heads.
        mlp_dims: MLP hidden dimension.
        dropout: Dropout probability.
        norm_first: Use pre-norm architecture.

    Example:
        >>> encoder = TransformerEncoder(dims=512, num_layers=6, num_heads=8)
        >>> x = mx.random.normal((2, 100, 512))
        >>> out = encoder(x)  # (2, 100, 512)
    """

    def __init__(
        self,
        dims: int,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.1,
        norm_first: bool = True
    ) -> None:
        super().__init__()
        self.layers = [
            TransformerEncoderBlock(dims, num_heads, mlp_dims, dropout, norm_first)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(dims) if norm_first else None

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder blocks.

    Args:
        dims: Model dimension.
        num_layers: Number of decoder blocks.
        num_heads: Number of attention heads.
        mlp_dims: MLP hidden dimension.
        dropout: Dropout probability.
        norm_first: Use pre-norm architecture.

    Example:
        >>> decoder = TransformerDecoder(dims=512, num_layers=6, num_heads=8)
        >>> x = mx.random.normal((2, 50, 512))
        >>> out, cache = decoder(x)
    """

    def __init__(
        self,
        dims: int,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.1,
        norm_first: bool = True
    ) -> None:
        super().__init__()
        self.layers = [
            TransformerDecoderBlock(dims, num_heads, mlp_dims, dropout, norm_first)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(dims) if norm_first else None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None
    ) -> Tuple[mx.array, list]:
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, c = layer(x, mask, layer_cache)
            new_cache.append(c)

        if self.norm is not None:
            x = self.norm(x)

        return x, new_cache


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Args:
        dims: Model dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.

    Example:
        >>> pe = PositionalEncoding(dims=512, max_len=1000)
        >>> x = mx.random.normal((2, 100, 512))
        >>> out = pe(x)  # Adds positional encoding
    """

    def __init__(self, dims: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encodings
        position = mx.arange(max_len)[:, None]
        div_term = mx.exp(mx.arange(0, dims, 2) * (-math.log(10000.0) / dims))

        pe = mx.zeros((max_len, dims))
        pe = pe.at[:, 0::2].add(mx.sin(position * div_term))
        pe = pe.at[:, 1::2].add(mx.cos(position * div_term))
        self._pe = pe[None, :, :]  # (1, max_len, dims)

    def __call__(self, x: mx.array) -> mx.array:
        """Add positional encoding to input."""
        x = x + self._pe[:, :x.shape[1], :]
        return self.dropout(x)


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image classification.

    Splits images into patches and processes them through a transformer encoder.

    Args:
        image_size: Input image size (assumes square images).
        patch_size: Size of image patches.
        num_classes: Number of output classes.
        dims: Model dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_dims: MLP hidden dimension.
        dropout: Dropout probability.
        channels: Number of input channels.

    Example:
        >>> vit = VisionTransformer(
        ...     image_size=224,
        ...     patch_size=16,
        ...     num_classes=1000,
        ...     dims=768,
        ...     depth=12,
        ...     num_heads=12
        ... )
        >>> x = mx.random.normal((2, 224, 224, 3))
        >>> logits = vit(x)  # (2, 1000)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.1,
        channels: int = 3
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, dims)

        # Class token and position embeddings
        self.cls_token = mx.zeros((1, 1, dims))
        self.pos_embed = mx.zeros((1, self.num_patches + 1, dims))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = TransformerEncoder(
            dims=dims,
            num_layers=depth,
            num_heads=num_heads,
            mlp_dims=mlp_dims,
            dropout=dropout
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Linear(dims, num_classes)
        )

    def _patchify(self, x: mx.array) -> mx.array:
        """Convert image to patches."""
        B, H, W, C = x.shape
        P = self.patch_size

        # Reshape to patches: (B, H/P, P, W/P, P, C) -> (B, num_patches, patch_dim)
        x = x.reshape(B, H // P, P, W // P, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, H/P, W/P, P, P, C)
        x = x.reshape(B, -1, P * P * C)  # (B, num_patches, patch_dim)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input images of shape (batch, height, width, channels).

        Returns:
            Logits of shape (batch, num_classes).
        """
        B = x.shape[0]

        # Create patches and embed
        x = self._patchify(x)  # (B, num_patches, patch_dim)
        x = self.patch_embed(x)  # (B, num_patches, dims)

        # Add class token
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, x.shape[-1]))
        x = mx.concatenate([cls_tokens, x], axis=1)  # (B, num_patches + 1, dims)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        x = self.encoder(x)

        # Classification from CLS token
        cls_output = x[:, 0]
        return self.head(cls_output)


# Convenience functions for common ViT variants
def ViT_Tiny(num_classes: int = 1000, image_size: int = 224, **kwargs) -> VisionTransformer:
    """ViT-Tiny: dims=192, depth=12, heads=3."""
    return VisionTransformer(
        image_size=image_size, num_classes=num_classes,
        dims=192, depth=12, num_heads=3, **kwargs
    )


def ViT_Small(num_classes: int = 1000, image_size: int = 224, **kwargs) -> VisionTransformer:
    """ViT-Small: dims=384, depth=12, heads=6."""
    return VisionTransformer(
        image_size=image_size, num_classes=num_classes,
        dims=384, depth=12, num_heads=6, **kwargs
    )


def ViT_Base(num_classes: int = 1000, image_size: int = 224, **kwargs) -> VisionTransformer:
    """ViT-Base: dims=768, depth=12, heads=12."""
    return VisionTransformer(
        image_size=image_size, num_classes=num_classes,
        dims=768, depth=12, num_heads=12, **kwargs
    )


def ViT_Large(num_classes: int = 1000, image_size: int = 224, **kwargs) -> VisionTransformer:
    """ViT-Large: dims=1024, depth=24, heads=16."""
    return VisionTransformer(
        image_size=image_size, num_classes=num_classes,
        dims=1024, depth=24, num_heads=16, **kwargs
    )


class GPT(nn.Module):
    """Simple GPT-style decoder-only transformer.

    Args:
        vocab_size: Size of vocabulary.
        max_seq_len: Maximum sequence length.
        dims: Model dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_dims: MLP hidden dimension.
        dropout: Dropout probability.

    Example:
        >>> model = GPT(vocab_size=50257, max_seq_len=1024, dims=768, num_layers=12)
        >>> tokens = mx.array([[1, 2, 3, 4, 5]])
        >>> logits = model(tokens)  # (1, 5, 50257)
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 1024,
        dims: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dims)
        self.pos_embed = nn.Embedding(max_seq_len, dims)
        self.dropout = nn.Dropout(dropout)

        self.decoder = TransformerDecoder(
            dims=dims,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dims=mlp_dims,
            dropout=dropout
        )

        self.head = nn.Linear(dims, vocab_size)

    def __call__(
        self,
        tokens: mx.array,
        cache: Optional[list] = None
    ) -> Tuple[mx.array, list]:
        """Forward pass.

        Args:
            tokens: Input token IDs of shape (batch, seq_len).
            cache: Optional KV cache for generation.

        Returns:
            Logits of shape (batch, seq_len, vocab_size) and updated cache.
        """
        B, L = tokens.shape

        # Get embeddings
        if cache is not None and len(cache) > 0 and cache[0] is not None:
            # During generation, we only need positions for new tokens
            start_pos = cache[0][0].shape[1]
            positions = mx.arange(start_pos, start_pos + L)
        else:
            positions = mx.arange(L)

        x = self.token_embed(tokens) + self.pos_embed(positions)
        x = self.dropout(x)

        # Decoder
        x, new_cache = self.decoder(x, cache=cache)

        # Output projection
        logits = self.head(x)

        return logits, new_cache

    def generate(
        self,
        prompt: mx.array,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> mx.array:
        """Generate tokens autoregressively.

        Args:
            prompt: Initial token IDs of shape (1, prompt_len).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: If set, only sample from top-k tokens.

        Returns:
            Generated token IDs of shape (1, prompt_len + max_tokens).
        """
        tokens = prompt
        cache = None

        for _ in range(max_tokens):
            logits, cache = self(tokens, cache)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # Top-k sampling
                top_values = mx.topk(next_logits, top_k)
                threshold = top_values[:, -1:]
                next_logits = mx.where(next_logits < threshold, float("-inf"), next_logits)

            probs = mx.softmax(next_logits, axis=-1)
            next_token = mx.random.categorical(probs, axis=-1)[:, None]
            tokens = mx.concatenate([tokens, next_token], axis=1)

        return tokens
