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

---

## Audit: FastEstimator vs FastMLX Comparison (December 2024)

This section documents the comprehensive comparison between the original FastEstimator
and the FastMLX port. Use this to understand design decisions and identify areas
for improvement.

### FastEstimator Design Philosophy (Reference)

FastEstimator follows these core principles that FastMLX should maintain:

1. **Composition over Inheritance**: Ops are composable sequences, not inheritance trees
2. **Separation of Concerns**: Pipeline (data), Network (computation), Estimator (orchestration)
3. **Declarative Configuration**: Lists of ops/traces describe the full training workflow
4. **Callback Pattern**: Traces hook into lifecycle without modifying core code
5. **Data Dictionary Pattern**: All data flows through shared dicts for op chaining
6. **Mode-Aware Execution**: Ops/traces know their execution context (train/eval/test/infer)

### Critical Discrepancies Identified

#### HIGH PRIORITY - Functional Issues

**1. EarlyStopping `restore_best_weights` is broken**
- Location: `fastmlx/trace/adapt.py:90-108`
- Issue: Tries to access `state['model']` which is never set by Estimator
- Fix: Store model reference in `__init__`, not from state

**2. UpdateOp recomputes forward pass unnecessarily**
- Location: `fastmlx/op/update_op.py:102-177`
- Issue: UpdateOp recomputes model forward pass to get gradients instead of using
  the already-computed outputs from ModelOp
- Impact: ~2x slower training than necessary
- Fix: Use MLX's tape-based autodiff to capture gradients during ModelOp's forward

**3. ModelOp doesn't handle multiple inputs properly**
- Location: `fastmlx/op/model_op.py:18-19`
- Issue: When `inputs=["x1", "x2"]`, passes list to model instead of unpacking
- Fix: Unpack list into positional arguments: `self.model(*data)` when data is list

**4. Loss ops inconsistent with target format**
- Location: Various loss ops in `fastmlx/op/`
- Issue: Some expect one-hot, some expect integer labels, inconsistent handling
- Fix: Each loss should document and handle both formats consistently

#### MEDIUM PRIORITY - Missing Features

**5. Missing Op class hierarchy (NumpyOp/TensorOp)**
- FastEstimator has:
  - `Op` (base)
  - `NumpyOp` (sample-level ops, numpy arrays, Pipeline)
  - `TensorOp` (batch-level ops, tensors, Network)
- FastMLX only has `Op` base class
- Impact: Less clear semantics about where ops belong
- Note: This is an acceptable simplification for MLX, but document clearly

**6. Missing mode negation syntax**
- FastEstimator supports `mode="!infer"` (run in all modes except infer)
- FastMLX only supports positive mode matching
- Fix: Add negation parsing to `Op.should_run()`

**7. Missing Trace mode filtering**
- FastEstimator traces have `mode` attribute to run only in specific modes
- FastMLX traces always run in all modes
- Impact: Can't have train-only or eval-only traces easily

**8. Missing `inputs`/`outputs` on Traces**
- FastEstimator traces declare what keys they read/write
- Enables automatic validation and monitoring
- FastMLX traces access keys dynamically without declaration

**9. Missing Pipeline.transform() utility**
- FastEstimator Pipeline has `transform()` for applying ops to single samples
- Useful for inference on individual samples outside training
- Fix: Add `transform(sample, mode="infer")` method

**10. Missing warmup validation**
- FastEstimator Estimator._warmup() validates graph before training
- Catches configuration errors early (missing keys, shape mismatches)
- FastMLX only catches errors at runtime

**11. Missing ds_id (dataset ID) support**
- FastEstimator ops can target specific datasets via `ds_id`
- Useful for multi-dataset training scenarios
- Lower priority for MLX use cases

#### LOW PRIORITY - API Polish

**12. Missing Data wrapper class for traces**
- FastEstimator uses `Data` class with `write_with_log(key, value)`
- Provides cleaner logging integration
- FastMLX passes raw dicts (simpler but less featured)

**13. Missing essential trace auto-injection**
- FastEstimator auto-injects TrainEssential, EvalEssential, Logger
- FastMLX requires manual trace addition
- Consider adding default traces with opt-out

**14. Summary not integrated with Estimator**
- FastEstimator.fit() returns Summary with full history
- FastMLX has Summary class but Estimator returns raw dict
- Fix: Return Summary from fit() instead of dict

**15. Network get_fe_loss_keys() pattern**
- FastEstimator TensorOps declare loss outputs via method
- FastMLX uses string matching ("loss" in key) - fragile
- Consider adding `is_loss` property to loss ops

### Recommended Fix Order

1. **ModelOp multiple inputs** - Simple fix, high impact
2. **EarlyStopping restore_best_weights** - Broken feature
3. **Trace mode filtering** - Add `mode` to Trace base class
4. **Pipeline.transform()** - Useful utility
5. **Mode negation** - Small enhancement
6. **UpdateOp efficiency** - Requires careful MLX autodiff work

### Code Quality Notes

**What's Done Well:**
- Clean separation of Pipeline/Network/Estimator
- Batch op marker pattern is elegant
- FilteredData sentinel for sample filtering
- DynamicBatch for variable-length sequences
- Comprehensive trace implementations (metrics, adaptation, IO)
- Good test coverage for core functionality
- AMP (mixed precision) support is well implemented

**What Needs Attention:**
- Some ops have minimal error handling
- Docstrings could be more comprehensive in some modules
- Type hints are inconsistent in places
- Some test files have incomplete coverage

### MLX-Specific Considerations

When working with this codebase, remember:

1. **Lazy Evaluation**: MLX ops are lazy. Call `mx.eval()` to force computation.
   This affects timing, memory profiling, and gradient computation.

2. **Unified Memory**: On Apple Silicon, CPU/GPU share memory. Don't worry about
   explicit device transfers like in PyTorch.

3. **Compilation**: MLX can compile functions for performance. UpdateOp has
   `compile` parameter but it's not fully utilized.

4. **Value and Grad**: Use `nn.value_and_grad()` pattern for efficient gradient
   computation. Current UpdateOp could be optimized.

5. **No Dynamic Shapes in Compiled Code**: MLX compiled functions need static shapes.
   Variable batch sizes may need recompilation.

### Testing Recommendations

When making changes:
1. Run `uv run pytest tests/` for full test suite
2. Pay special attention to integration tests (test_mnist_training.py, etc.)
3. Test both train and eval modes
4. Test with mixed precision enabled
5. Verify gradient flow with simple models

### Future Architecture Considerations

If doing a major refactor, consider:

1. **Unified Op base with type hints**: Single Op class but with clear typing
   for expected array types (numpy vs mx.array)

2. **Trace registry pattern**: Auto-discover and validate trace dependencies

3. **Deferred execution model**: Queue ops and optimize execution order

4. **Plugin system for backends**: Abstract MLX specifics for potential future
   backends (CUDA direct, Metal Performance Shaders, etc.)
