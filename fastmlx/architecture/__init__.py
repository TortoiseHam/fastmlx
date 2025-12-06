"""Reference neural network architectures for FastMLX."""

from .lenet import LeNet
from .resnet9 import ResNet9
from .unet import UNet, AttentionUNet
from .wideresnet import (
    WideResNet,
    WideResNet16_8,
    WideResNet28_10,
    WideResNet40_4,
)
from .transformer import (
    MultiHeadAttention,
    TransformerEncoderBlock,
    TransformerDecoderBlock,
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,
    VisionTransformer,
    ViT_Tiny,
    ViT_Small,
    ViT_Base,
    ViT_Large,
    GPT,
)
from .siamese import SiameseNetwork, SiameseEncoder
from .autoencoder import Autoencoder, VAE

__all__ = [
    # Classification - CNN
    "LeNet",
    "ResNet9",
    "WideResNet",
    "WideResNet16_8",
    "WideResNet28_10",
    "WideResNet40_4",
    # Segmentation
    "UNet",
    "AttentionUNet",
    # Transformer Components
    "MultiHeadAttention",
    "TransformerEncoderBlock",
    "TransformerDecoderBlock",
    "TransformerEncoder",
    "TransformerDecoder",
    "PositionalEncoding",
    # Vision Transformers
    "VisionTransformer",
    "ViT_Tiny",
    "ViT_Small",
    "ViT_Base",
    "ViT_Large",
    # Language Models
    "GPT",
    # Metric Learning
    "SiameseNetwork",
    "SiameseEncoder",
    # Generative Models
    "Autoencoder",
    "VAE",
]
