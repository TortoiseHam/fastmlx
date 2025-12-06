"""Operation classes used within FastMLX."""

from .op import Op

# Utility Ops
from .lambda_op import LambdaOp, Delete, RemoveIf, Reshape, Cast, Squeeze, Clip

# Basic Preprocessing Ops
from .expand_dims import ExpandDims
from .minmax import Minmax
from .normalize import Normalize
from .pad_if_needed import PadIfNeeded
from .onehot import Onehot
from .sometimes import Sometimes

# Geometric Augmentation Ops
from .random_crop import RandomCrop
from .horizontal_flip import HorizontalFlip
from .vertical_flip import VerticalFlip
from .rotate import Rotate, Rotate90
from .resize import Resize, RandomResizedCrop, CenterCrop

# Pixel-level Augmentation Ops
from .coarse_dropout import CoarseDropout
from .gaussian_blur import GaussianBlur
from .gaussian_noise import GaussianNoise
from .brightness_contrast import Brightness, Contrast, RandomBrightnessContrast
from .color_jitter import ColorJitter, ToGray
from .cutout import Cutout, GridMask, MixUp
from .random_erasing import RandomErasing, GridDropout, ChannelDropout
from .elastic_transform import ElasticTransform, PerspectiveTransform, ShearTransform

# Loss Ops
from .cross_entropy import CrossEntropy
from .mean_squared_error import MeanSquaredError
from .l1_loss import L1Loss
from .focal_loss import FocalLoss
from .dice_loss import DiceLoss
from .hinge_loss import HingeLoss
from .smooth_l1_loss import SmoothL1Loss
from .label_smoothing import LabelSmoothingCrossEntropy, WeightedCrossEntropy
from .metric_loss import TripletLoss, ContrastiveLoss, CenterLoss, CosineSimilarityLoss

# Model Ops
from .model_op import ModelOp
from .update_op import UpdateOp

__all__ = [
    # Base
    "Op",
    # Utility Ops
    "LambdaOp",
    "Delete",
    "RemoveIf",
    "Reshape",
    "Cast",
    "Squeeze",
    "Clip",
    # Basic Preprocessing
    "ExpandDims",
    "Minmax",
    "Normalize",
    "PadIfNeeded",
    "Onehot",
    "Sometimes",
    # Geometric Augmentation
    "RandomCrop",
    "HorizontalFlip",
    "VerticalFlip",
    "Rotate",
    "Rotate90",
    "Resize",
    "RandomResizedCrop",
    "CenterCrop",
    # Pixel-level Augmentation
    "CoarseDropout",
    "GaussianBlur",
    "GaussianNoise",
    "Brightness",
    "Contrast",
    "RandomBrightnessContrast",
    "ColorJitter",
    "ToGray",
    "Cutout",
    "GridMask",
    "MixUp",
    "RandomErasing",
    "GridDropout",
    "ChannelDropout",
    "ElasticTransform",
    "PerspectiveTransform",
    "ShearTransform",
    # Loss functions
    "CrossEntropy",
    "MeanSquaredError",
    "L1Loss",
    "FocalLoss",
    "DiceLoss",
    "HingeLoss",
    "SmoothL1Loss",
    "LabelSmoothingCrossEntropy",
    "WeightedCrossEntropy",
    "TripletLoss",
    "ContrastiveLoss",
    "CenterLoss",
    "CosineSimilarityLoss",
    # Model operations
    "ModelOp",
    "UpdateOp",
]
