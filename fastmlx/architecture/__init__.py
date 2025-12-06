"""Reference neural network architectures for FastMLX."""

from .autoencoder import VAE, Autoencoder
from .gan import (
    DCGANDiscriminator,
    DCGANGenerator,
    SimpleDiscriminator,
    SimpleGenerator,
)
from .lenet import LeNet
from .resnet9 import ResNet9
from .siamese import SiameseEncoder, SiameseNetwork
from .transformer import (
    GPT,
    MultiHeadAttention,
    PositionalEncoding,
    TransformerDecoder,
    TransformerDecoderBlock,
    TransformerEncoder,
    TransformerEncoderBlock,
    VisionTransformer,
    ViT_Base,
    ViT_Large,
    ViT_Small,
    ViT_Tiny,
)
from .unet import AttentionUNet, UNet
from .wideresnet import (
    WideResNet,
    WideResNet16_8,
    WideResNet28_10,
    WideResNet40_4,
)

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
    "DCGANGenerator",
    "DCGANDiscriminator",
    "SimpleGenerator",
    "SimpleDiscriminator",
]
