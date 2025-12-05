# FastMLX

A deep learning framework based on [MLX](https://github.com/ml-explore/mlx) with an API inspired by [FastEstimator](https://github.com/fastestimator/fastestimator).

## Installation

### Prerequisites

- Python 3.10+
- macOS with Apple Silicon OR Linux with CUDA

### macOS (Apple Silicon)

```bash
# Install UV (recommended package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/TortoiseHam/fastmlx.git
cd fastmlx

# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install with all extras
uv pip install -e ".[all,dev]"
```

### Linux (CUDA)

MLX 0.30+ supports CUDA on Linux.

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/TortoiseHam/fastmlx.git
cd fastmlx
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

## Quick Start

```python
import fastmlx as fm
from fastmlx.op import Minmax, CrossEntropy, ModelOp, UpdateOp
from fastmlx.trace import Accuracy
from fastmlx.architecture import LeNet

# Build model
model = fm.build(LeNet, optimizer_fn="adam")

# Create pipeline and network
pipeline = fm.Pipeline(train_data=train_ds, eval_data=eval_ds, batch_size=32,
                       ops=[Minmax(inputs="x", outputs="x")])

network = fm.Network([
    ModelOp(model=model, inputs="x", outputs="y_pred"),
    CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    UpdateOp(model=model, loss_name="ce")
])

# Train
estimator = fm.Estimator(pipeline=pipeline, network=network, epochs=10,
                         traces=[Accuracy(true_key="y", pred_key="y_pred")])
estimator.fit()
```

## Examples

See `fastmlx/apphub/` for complete examples:
- `mnist.py` - MNIST with LeNet
- `cifar10.py` - CIFAR-10 with ResNet9

## Running Tests

FastMLX has comprehensive tests covering all major components.

### Quick Test Run

```bash
# Run all tests
uv run pytest tests/ -v

# Run tests with coverage report
uv run pytest tests/ --cov=fastmlx --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_network.py -v

# Run tests matching a pattern
uv run pytest tests/ -k "test_accuracy" -v
```

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_network.py` | Network forward pass, op chaining |
| `test_pipeline.py` | Data pipeline, batching, ops |
| `test_dataset.py` | MLXDataset, CSVDataset, generators |
| `test_ops_preprocessing.py` | Minmax, Normalize, Onehot, etc. |
| `test_ops_loss.py` | CrossEntropy, MSE, Focal, Dice |
| `test_ops_augmentation.py` | Flip, Rotate, Noise, Cutout |
| `test_traces.py` | Accuracy, F1, EarlyStopping |
| `test_architecture.py` | LeNet, ResNet9, UNet, ViT, GPT |
| `test_schedule_full.py` | Cosine, linear, warmup schedules |
| `test_search.py` | GridSearch, RandomSearch |
| `test_summary.py` | Experiment tracking |
| `test_backend.py` | Device, memory, dtype utilities |

### Running Tests by Component

```bash
# Core components
uv run pytest tests/test_network.py tests/test_pipeline.py -v

# All ops
uv run pytest tests/test_ops_*.py -v

# Architectures
uv run pytest tests/test_architecture.py -v

# Data loading (important for understanding Pipeline behavior)
uv run pytest tests/test_dataset.py tests/test_pipeline.py -v
```

### Writing New Tests

Tests use Python's `unittest` framework. Example:

```python
import unittest
import mlx.core as mx
from fastmlx.op import MyNewOp

class TestMyNewOp(unittest.TestCase):
    def test_forward(self):
        op = MyNewOp("x", "y")
        data = mx.zeros((4, 28, 28, 1))
        result = op.forward(data, {})
        self.assertEqual(result.shape, (4, 28, 28, 1))

if __name__ == "__main__":
    unittest.main()
```

## License

MIT
