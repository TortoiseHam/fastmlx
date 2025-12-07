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

**1. EarlyStopping `restore_best_weights` is broken** ✅ FIXED
- Location: `fastmlx/trace/adapt.py`
- Issue: Tried to access `state['model']` which was never set
- Fix: Now takes `model` parameter in `__init__`, stores/restores weights properly

**2. UpdateOp recomputes forward pass unnecessarily** ✅ FIXED
- Location: `fastmlx/network.py`
- Issue: UpdateOp recomputed forward pass for gradients
- Impact: Was ~2x slower than necessary
- Fix: Network now composes all ops into single traced function for `nn.value_and_grad()`
  See "MLX Autodiff Architecture" section above.

**3. ModelOp doesn't handle multiple inputs properly** ✅ FIXED
- Location: `fastmlx/op/model_op.py`
- Issue: When `inputs=["x1", "x2"]`, passed list instead of unpacking
- Fix: Now unpacks: `self.model(*data)` when data is list

**4. Loss ops inconsistent with target format** ✅ FIXED
- All loss ops now inherit from `LossOp` base class
- Added `is_loss` property for explicit identification
- Each loss handles both one-hot and integer label formats

#### MEDIUM PRIORITY - Missing Features

**5. Missing Op class hierarchy (NumpyOp/TensorOp)**
- FastEstimator has:
  - `Op` (base)
  - `NumpyOp` (sample-level ops, numpy arrays, Pipeline)
  - `TensorOp` (batch-level ops, tensors, Network)
- FastMLX only has `Op` base class
- Impact: Less clear semantics about where ops belong
- Note: This is an acceptable simplification for MLX, but document clearly

**6. Missing mode negation syntax** ✅ FIXED
- Added `parse_modes()` in `Op` to support `mode="!infer"` syntax
- Works for both Ops and Traces

**7. Missing Trace mode filtering** ✅ FIXED
- Added `mode` parameter to `Trace` base class
- All traces now call `super().__init__(mode=mode)`
- Estimator uses `_trace_should_run()` to filter by mode

**8. Missing `inputs`/`outputs` on Traces** ✅ FIXED
- Added `inputs` and `outputs` parameters to `Trace` base class
- All metric/io/adapt traces now declare their inputs/outputs

**9. Missing Pipeline.transform() utility** ✅ FIXED
- Added `Pipeline.transform(sample, mode="infer")` method
- Applies pipeline ops to single samples for inference

**10. Missing warmup validation** ✅ FIXED
- Added `Estimator._warmup()` method
- Runs single batch through network before training to catch errors early
- Validates both train and eval modes

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

**14. Summary not integrated with Estimator** ✅ FIXED
- `Estimator.fit()` now returns `Summary` object
- Metrics are recorded per epoch with mode tracking
- Added `experiment_name` parameter

**15. Network get_loss_keys() pattern** ✅ FIXED
- Added `LossOp` base class with `is_loss = True` property
- All loss ops now inherit from `LossOp`
- `Network.get_loss_keys()` uses `is_loss` property with fallback to string matching

### Fix Status Summary

All 15 identified issues have been addressed:
- ✅ 1-4: HIGH PRIORITY issues fixed
- ✅ 5-11: MEDIUM PRIORITY issues fixed (except ds_id which is low priority for MLX)
- ✅ 12-15: LOW PRIORITY issues fixed

Remaining low-priority items:
- ds_id support (multi-dataset targeting) - not critical for MLX use cases
- Data wrapper class - current dict approach works fine
- Essential trace auto-injection - users can add traces explicitly

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

3. **Compilation**: MLX can compile functions for performance.

4. **No Dynamic Shapes in Compiled Code**: MLX compiled functions need static shapes.
   Variable batch sizes may need recompilation.

### MLX Autodiff Architecture (CRITICAL)

**Why MLX is different from PyTorch:**

PyTorch uses tape-based autodiff where tensors carry computation history:
```python
y_pred = model(x)           # Tensor retains graph
loss = loss_fn(y_pred, y)   # Connected to graph
loss.backward()             # Walks existing graph
```

MLX uses functional autodiff (like JAX) - no computation history in arrays:
```python
def loss_fn(model):
    return compute_loss(model(x), y)

loss, grads = nn.value_and_grad(model, loss_fn)(model)  # Must trace function
```

**Solution: Network composes ops at runtime**

The `Network` class automatically handles this:
1. Analyzes op graph to find ModelOps, LossOps, and UpdateOps
2. In training mode, builds a single composed function containing all forward/loss ops
3. Passes that function to `nn.value_and_grad()` for efficient gradient computation
4. Applies gradients via UpdateOp configuration (clipping, accumulation, etc.)

This means users keep the familiar FastEstimator API:
```python
network = Network([
    ModelOp(model=model, inputs="x", outputs="y_pred"),
    CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    UpdateOp(model=model, loss_name="ce")
])
```

And the Network automatically composes into efficient single-pass execution.

**Multi-model support**: Works with GANs, teacher-student, etc. - Network handles
computing gradients for each model separately.

**Key implementation**: See `Network._run_training()` and `Network._build_forward_loss_fn()`

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
