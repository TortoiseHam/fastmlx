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

```bash
pytest tests/
```

## License

MIT
