"""Operation classes used within FastMLX."""

# Batch and filtering
from .batch import Batch, DynamicBatch
from .brightness_contrast import Brightness, Contrast, RandomBrightnessContrast

# Pixel-level Augmentation Ops
from .coarse_dropout import CoarseDropout
from .color_jitter import ColorJitter, ToGray

# Loss Ops
from .cross_entropy import CrossEntropy
from .cutout import CutMix, Cutout, GridMask, MixUp
from .dice_loss import DiceLoss
from .elastic_transform import ElasticTransform, PerspectiveTransform, ShearTransform

# Basic Preprocessing Ops
from .expand_dims import ExpandDims
from .filtered_data import FilteredData
from .focal_loss import FocalLoss
from .gaussian_blur import GaussianBlur
from .gaussian_noise import GaussianNoise
from .hinge_loss import HingeLoss
from .horizontal_flip import HorizontalFlip
from .l1_loss import L1Loss
from .label_smoothing import LabelSmoothingCrossEntropy, WeightedCrossEntropy

# Utility Ops
from .lambda_op import Cast, Clip, Delete, LambdaOp, RemoveIf, Reshape, Squeeze
from .mean_squared_error import MeanSquaredError
from .metric_loss import CenterLoss, ContrastiveLoss, CosineSimilarityLoss, TripletLoss
from .minmax import Minmax

# Model Ops
from .model_op import ModelOp
from .normalize import Normalize
from .onehot import Onehot
from .op import LossOp, Op
from .pad_if_needed import PadIfNeeded

# Geometric Augmentation Ops
from .random_crop import RandomCrop
from .random_erasing import ChannelDropout, GridDropout, RandomErasing
from .resize import CenterCrop, RandomResizedCrop, Resize
from .rotate import Rotate, Rotate90
from .smooth_l1_loss import SmoothL1Loss
from .sometimes import Sometimes
from .super_loss import ConfidenceWeightedLoss, GradientWeightedLoss, SuperLoss
from .update_op import UpdateOp
from .vertical_flip import VerticalFlip

__all__ = [
    # Base
    "Op",
    "LossOp",
    # Batch and filtering
    "Batch",
    "DynamicBatch",
    "FilteredData",
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
    "CutMix",
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
    "SuperLoss",
    "ConfidenceWeightedLoss",
    "GradientWeightedLoss",
    # Model operations
    "ModelOp",
    "UpdateOp",
]
