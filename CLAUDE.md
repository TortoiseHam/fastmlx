# Claude Development Notes for FastMLX

## Project Overview

FastMLX is a port of FastEstimator to the MLX framework. The goal is to provide a high-level deep learning API that works seamlessly on Apple Silicon (Metal) and Linux (CUDA via MLX 0.30.0+).

## Architecture

The codebase follows FastEstimator's design patterns:

```
fastmlx/
├── __init__.py          # Main exports, version
├── pipeline.py          # Data pipeline with ops
├── network.py           # Model forward/backward
├── estimator.py         # Training loop orchestration
├── op/                  # Operations (transforms, losses, model ops)
├── trace/               # Callbacks (metrics, logging, checkpointing)
├── architecture/        # Neural network architectures
├── dataset/             # Dataset classes
├── schedule/            # Learning rate schedules
├── search/              # Hyperparameter search
├── summary/             # Experiment tracking
└── backend/             # Device/memory/dtype utilities
```

## Key Design Patterns

### Op (Operation)
- Base class in `op/op.py`
- Has `inputs`, `outputs`, `mode` attributes
- Override `forward()` method for execution
- Used in Pipeline (data ops) and Network (model ops)

### Trace
- Base class in `trace/trace.py`
- Callbacks with `on_begin`, `on_batch_end`, `on_epoch_end`, etc.
- Access data via `data` dict passed to methods

### Estimator Training Loop
```python
estimator = Estimator(
    pipeline=pipeline,   # Handles data loading and augmentation
    network=network,     # Handles forward/backward passes
    epochs=10,
    traces=[...]         # Callbacks for metrics, logging, etc.
)
estimator.fit()
```

## Porting from FastEstimator

When porting features from FastEstimator:

1. **Check the original**: Look at https://github.com/fastestimator/fastestimator
2. **Adapt for MLX**: Replace TensorFlow/PyTorch ops with MLX equivalents
3. **Keep API similar**: Users familiar with FastEstimator should feel at home
4. **Use `mx.array`**: MLX arrays instead of TF tensors or PyTorch tensors

## MLX Specifics

- **Lazy evaluation**: Operations are queued until `mx.eval()` is called
- **Unified memory**: CPU and GPU share memory on Apple Silicon
- **Device selection**: `mx.set_default_device(mx.gpu)` or `mx.cpu`
- **CUDA support**: MLX 0.30.0+ supports CUDA on Linux

## Testing

Run tests with:
```bash
uv run pytest tests/
```

## Common Tasks

### Adding a new Op
1. Create file in `fastmlx/op/` (e.g., `my_op.py`)
2. Inherit from `Op` or `TensorOp`
3. Implement `forward(self, data, state)` method
4. Export in `fastmlx/op/__init__.py`

### Adding a new Trace
1. Create in `fastmlx/trace/` (adapt.py, io.py, or metric.py)
2. Inherit from `Trace`
3. Implement relevant hooks (`on_epoch_end`, etc.)
4. Export in `fastmlx/trace/__init__.py`

### Adding a new Architecture
1. Create file in `fastmlx/architecture/`
2. Inherit from `mlx.nn.Module`
3. Export in `fastmlx/architecture/__init__.py`

## What's Been Ported (as of v0.2.0)

### Ops
- **Tensor**: Minmax, Normalize, Onehot, ExpandDims, Squeeze, etc.
- **Loss**: CrossEntropy, MeanSquaredError, FocalLoss, DiceLoss, etc.
- **Augmentation**: HorizontalFlip, VerticalFlip, Rotate, GaussianBlur, etc.
- **Model**: ModelOp, UpdateOp

### Traces
- **Metric**: Accuracy, Precision, Recall, F1Score, ConfusionMatrix, Dice, MCC
- **Adapt**: EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, WarmupScheduler
- **IO**: BestModelSaver, ModelSaver, CSVLogger, ProgressLogger, Timer

### Architectures
- **CNN**: LeNet, ResNet9, WideResNet variants, UNet, AttentionUNet
- **Transformer**: ViT (Tiny/Small/Base/Large), GPT, encoder/decoder blocks

### Other
- **Schedule**: cosine_decay, linear_decay, step_decay, one_cycle, etc.
- **Search**: GridSearch, RandomSearch, GoldenSection
- **Summary**: Summary (metrics history), Experiment (full tracking)
- **Backend**: Device management, memory utils, dtype helpers, seeding

## What Still Needs Porting

Priority items from FastEstimator:
- [ ] More dataset types (SiameseDirDataset, etc.)
- [ ] TensorBoard integration
- [ ] Traceability reports
- [ ] More augmentation ops (elastic transform, etc.)
- [ ] Distributed training support
- [ ] More loss functions (SuperLoss, etc.)
- [ ] Image/batch visualization improvements

## Development Commands

```bash
# Install in dev mode
uv pip install -e ".[dev]"

# Run linter
uv run ruff check fastmlx/

# Run type checker
uv run mypy fastmlx/

# Run tests
uv run pytest tests/ -v
```

## Git Workflow

- Main branch: `main`
- Feature branches: `claude/<description>-<session-id>`
- Always create PR for review before merging
