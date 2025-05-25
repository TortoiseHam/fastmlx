# fastmlx

A lightweight machine learning framework based on [MLX](https://github.com/ml-explore/mlx) with an API inspired by [FastEstimator](https://github.com/fastestimator/fastestimator).

This repository contains a minimal implementation sufficient to run the MNIST example in `apphub/mnist.py` and a simple CIFAR-10 variant in `apphub/cifar10.py`.

## Installation

Install `fastmlx` using `pip`:

```bash
pip install -e .
```

## Running Tests

The unit tests use Python's ``unittest`` framework. Run all tests with:

```bash
python -m unittest discover -s tests
```
