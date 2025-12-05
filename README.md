# FastMLX

A deep learning framework based on [MLX](https://github.com/ml-explore/mlx) with an API inspired by [FastEstimator](https://github.com/fastestimator/fastestimator).

FastMLX provides a high-level, intuitive API for building and training neural networks on Apple Silicon, combining the performance benefits of MLX with the clean design patterns of FastEstimator.

## Features

- **FastEstimator-style API**: Familiar Pipeline, Network, Estimator pattern
- **MLX Backend**: Native Apple Silicon acceleration
- **Comprehensive Ops**: 30+ operations for data augmentation and loss computation
- **Flexible Traces**: Metrics, callbacks, and logging utilities
- **Pre-built Architectures**: LeNet, ResNet9, WideResNet, UNet, and more
- **Learning Rate Schedules**: Cosine decay, warmup, one-cycle, and custom schedulers
- **Dataset Utilities**: CSV, directory-based, and generator datasets

## Installation

Install `fastmlx` using pip:

```bash
pip install -e .
```

Or with UV (recommended):

```bash
uv pip install -e .
```

For development dependencies:

```bash
uv pip install -e ".[dev]"
```

## Quick Start

```python
import fastmlx as fm
from fastmlx.op import Minmax, CrossEntropy, ModelOp, UpdateOp
from fastmlx.trace import Accuracy, BestModelSaver, LRScheduler
from fastmlx.architecture import LeNet
from fastmlx.schedule import cosine_decay
from fastmlx.dataset.data import mnist

# Load data
train_data, eval_data = mnist.load_data()

# Create pipeline with preprocessing
pipeline = fm.Pipeline(
    train_data=train_data,
    eval_data=eval_data,
    batch_size=32,
    ops=[Minmax(inputs="x", outputs="x")]
)

# Build model with optimizer
model = fm.build(LeNet, optimizer_fn="adam")

# Create network with forward pass and loss
network = fm.Network([
    ModelOp(model=model, inputs="x", outputs="y_pred"),
    CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    UpdateOp(model=model, loss_name="ce")
])

# Configure traces for metrics and callbacks
traces = [
    Accuracy(true_key="y", pred_key="y_pred"),
    BestModelSaver(model=model, save_dir="/tmp/model", metric="accuracy"),
    LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, 1000, 0.001))
]

# Train
estimator = fm.Estimator(
    pipeline=pipeline,
    network=network,
    epochs=10,
    traces=traces
)
estimator.fit()
estimator.test()
```

## Components

### Operations (Ops)

Operations transform data in the pipeline or network:

**Preprocessing & Augmentation:**
- `Minmax`, `Normalize` - Normalization
- `RandomCrop`, `CenterCrop`, `Resize` - Spatial transforms
- `HorizontalFlip`, `VerticalFlip`, `Rotate` - Geometric augmentation
- `GaussianBlur`, `GaussianNoise` - Blur and noise
- `Brightness`, `Contrast`, `ColorJitter` - Color augmentation
- `Cutout`, `GridMask`, `MixUp` - Advanced augmentation

**Loss Functions:**
- `CrossEntropy`, `FocalLoss` - Classification losses
- `MeanSquaredError`, `L1Loss`, `SmoothL1Loss` - Regression losses
- `DiceLoss` - Segmentation loss
- `HingeLoss` - Margin-based loss

**Utility:**
- `LambdaOp` - Custom transformations
- `Sometimes` - Probabilistic op wrapper
- `ModelOp`, `UpdateOp` - Model forward pass and gradient updates

### Traces

Traces provide hooks into the training loop:

**Metrics:**
- `Accuracy`, `Precision`, `Recall`, `F1Score`
- `ConfusionMatrix`, `MCC`, `Dice`
- `LossMonitor`

**Adaptation:**
- `LRScheduler`, `WarmupScheduler`
- `EarlyStopping`, `ReduceLROnPlateau`
- `TerminateOnNaN`

**I/O:**
- `BestModelSaver`, `ModelSaver`
- `CSVLogger`, `ProgressLogger`, `Timer`

### Architectures

Pre-built neural network architectures:

- **Classification**: `LeNet`, `ResNet9`, `WideResNet`, `WideResNet28_10`
- **Segmentation**: `UNet`, `AttentionUNet`

### Learning Rate Schedules

```python
from fastmlx.schedule import (
    cosine_decay,
    linear_decay,
    step_decay,
    exponential_decay,
    warmup_cosine_decay,
    one_cycle,
    EpochScheduler,
    RepeatScheduler
)
```

### Datasets

```python
from fastmlx.dataset import (
    MLXDataset,          # In-memory array dataset
    CSVDataset,          # Load from CSV files
    DirDataset,          # Load from directory
    LabeledDirDataset,   # Labeled directory structure
    GeneratorDataset,    # Generator-based lazy loading
    BatchDataset,        # Batch wrapper
    CombinedDataset,     # Concatenate datasets
    InterleaveDataset    # Interleave datasets
)
```

## Examples

See the `apphub/` directory for complete examples:

- `apphub/mnist.py` - MNIST classification with LeNet
- `apphub/cifar10.py` - CIFAR-10 with ResNet9 and data augmentation

## Running Tests

```bash
# Using pytest
pytest tests/

# Or with unittest
python -m unittest discover -s tests
```

## Requirements

- Python >= 3.11
- macOS with Apple Silicon (M1/M2/M3)
- MLX >= 0.20.0

## License

MIT License
