"""Time Series Forecasting example using :mod:`fastmlx`.

Demonstrates training neural networks for time series prediction
using synthetic data. Shows:
- Windowed data preparation
- LSTM/Transformer-style sequence modeling
- Multi-step ahead forecasting

Applicable to:
- Stock price prediction
- Weather forecasting
- Energy demand prediction
- Sensor data analysis
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import fastmlx as fe
from fastmlx.dataset import MLXDataset
from fastmlx.op import MeanSquaredError, ModelOp, UpdateOp, Op
from fastmlx.schedule import cosine_decay
from fastmlx.trace.base import Trace
from fastmlx.trace.io import ModelSaver
from fastmlx.trace.adapt import LRScheduler


class LSTMForecaster(nn.Module):
    """LSTM-based time series forecaster.

    Args:
        input_dim: Number of input features per timestep.
        hidden_dim: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        output_dim: Number of output features (forecast dimension).
        forecast_horizon: Number of future steps to predict.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        forecast_horizon: int = 1
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        # Stack of LSTM cells (simplified - MLX doesn't have built-in LSTM)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = [
            nn.Linear(hidden_dim, hidden_dim * 4)  # Gates: i, f, g, o
            for _ in range(num_layers)
        ]
        self.output_proj = nn.Linear(hidden_dim, output_dim * forecast_horizon)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, input_dim).

        Returns:
            Predictions of shape (batch, forecast_horizon, output_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Simple RNN-style processing (simplified LSTM)
        for layer in self.layers:
            gates = layer(h)  # (batch, seq_len, hidden_dim * 4)
            i, f, g, o = mx.split(gates, 4, axis=-1)
            i = mx.sigmoid(i)
            f = mx.sigmoid(f)
            g = mx.tanh(g)
            o = mx.sigmoid(o)

            # Cell state update (simplified)
            h = o * mx.tanh(f * h + i * g)

        # Take last timestep and project to output
        h_last = h[:, -1, :]  # (batch, hidden_dim)
        out = self.output_proj(h_last)  # (batch, output_dim * forecast_horizon)

        return out.reshape(batch_size, self.forecast_horizon, -1)


class TransformerForecaster(nn.Module):
    """Transformer-based time series forecaster.

    Uses self-attention for capturing long-range dependencies.

    Args:
        input_dim: Number of input features per timestep.
        d_model: Transformer model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        output_dim: Number of output features.
        forecast_horizon: Number of future steps to predict.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        output_dim: int = 1,
        forecast_horizon: int = 1,
        max_seq_len: int = 100
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = [
            nn.TransformerEncoderLayer(d_model, num_heads)
            for _ in range(num_layers)
        ]

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim * forecast_horizon)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        batch_size, seq_len, _ = x.shape

        # Project and add positional encoding
        h = self.input_proj(x)
        positions = mx.arange(seq_len)
        h = h + self.pos_embed(positions)

        # Transformer layers
        for layer in self.layers:
            h = layer(h)

        # Take last timestep and project
        h_last = h[:, -1, :]
        out = self.output_proj(h_last)

        return out.reshape(batch_size, self.forecast_horizon, -1)


class MLPForecaster(nn.Module):
    """Simple MLP forecaster for time series.

    Flattens the input window and uses fully-connected layers.
    """

    def __init__(
        self,
        input_dim: int = 1,
        window_size: int = 24,
        hidden_dim: int = 128,
        output_dim: int = 1,
        forecast_horizon: int = 1
    ) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim

        flat_input = input_dim * window_size
        self.net = nn.Sequential(
            nn.Linear(flat_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * forecast_horizon),
        )

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        out = self.net(x_flat)
        return out.reshape(batch_size, self.forecast_horizon, self.output_dim)


def generate_synthetic_timeseries(
    num_samples: int = 10000,
    seq_length: int = 100,
    num_features: int = 1,
    seed: int = 42
) -> np.ndarray:
    """Generate synthetic time series with trend, seasonality, and noise.

    Args:
        num_samples: Total number of time points.
        seq_length: Not used, kept for API compatibility.
        num_features: Number of features (channels).
        seed: Random seed.

    Returns:
        Time series array of shape (num_samples, num_features).
    """
    np.random.seed(seed)

    t = np.arange(num_samples)

    # Trend component
    trend = 0.01 * t

    # Seasonal components (multiple frequencies)
    seasonal1 = 10 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    seasonal2 = 5 * np.sin(2 * np.pi * t / 168)  # Weekly pattern

    # Noise
    noise = np.random.randn(num_samples) * 2

    # Combine
    series = trend + seasonal1 + seasonal2 + noise

    # Reshape for multiple features
    if num_features > 1:
        series = np.column_stack([
            series + np.random.randn(num_samples) * 0.5
            for _ in range(num_features)
        ])
    else:
        series = series.reshape(-1, 1)

    return series.astype(np.float32)


def create_sequences(
    data: np.ndarray,
    window_size: int = 24,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create input-output sequences for supervised learning.

    Args:
        data: Time series of shape (num_samples, num_features).
        window_size: Number of past timesteps for input.
        forecast_horizon: Number of future timesteps to predict.

    Returns:
        Tuple of (inputs, targets) arrays.
    """
    num_samples = len(data) - window_size - forecast_horizon + 1

    inputs = np.zeros((num_samples, window_size, data.shape[1]))
    targets = np.zeros((num_samples, forecast_horizon, data.shape[1]))

    for i in range(num_samples):
        inputs[i] = data[i:i + window_size]
        targets[i] = data[i + window_size:i + window_size + forecast_horizon]

    return inputs, targets


class MAE(Trace):
    """Mean Absolute Error metric."""

    def __init__(self, pred_key: str = "y_pred", true_key: str = "y") -> None:
        self.pred_key = pred_key
        self.true_key = true_key
        self.total_error = 0.0
        self.count = 0

    def on_epoch_begin(self, state):
        self.total_error = 0.0
        self.count = 0

    def on_batch_end(self, batch, state):
        pred = batch[self.pred_key]
        true = batch[self.true_key]
        mae = float(mx.mean(mx.abs(pred - true)).item())
        self.total_error += mae
        self.count += 1

    def on_epoch_end(self, state):
        state['metrics']['mae'] = self.total_error / max(1, self.count)


def get_estimator(
    epochs: int = 50,
    batch_size: int = 32,
    window_size: int = 24,
    forecast_horizon: int = 1,
    model_type: str = "mlp",
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create time series forecasting estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        window_size: Input window size (look-back period).
        forecast_horizon: Number of steps ahead to predict.
        model_type: 'mlp', 'lstm', or 'transformer'.
        save_dir: Directory to save model.

    Returns:
        Configured Estimator ready for training.
    """
    # Generate synthetic data
    print("Generating synthetic time series...")
    series = generate_synthetic_timeseries(num_samples=5000)

    # Create sequences
    X, y = create_sequences(series, window_size, forecast_horizon)

    # Normalize
    mean = X.mean()
    std = X.std() + 1e-8
    X = (X - mean) / std
    y = (y - mean) / std

    # Split train/eval
    split = int(len(X) * 0.8)
    train_X, train_y = X[:split], y[:split]
    eval_X, eval_y = X[split:], y[split:]

    print(f"Training samples: {len(train_X)}, Eval samples: {len(eval_X)}")

    train_data = MLXDataset({
        "x": mx.array(train_X),
        "y": mx.array(train_y)
    })
    eval_data = MLXDataset({
        "x": mx.array(eval_X),
        "y": mx.array(eval_y)
    })

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[],  # Data is already preprocessed
    )

    # Select model
    input_dim = series.shape[1]
    output_dim = input_dim

    if model_type == "lstm":
        model_fn = lambda: LSTMForecaster(
            input_dim=input_dim, hidden_dim=64, num_layers=2,
            output_dim=output_dim, forecast_horizon=forecast_horizon
        )
    elif model_type == "transformer":
        model_fn = lambda: TransformerForecaster(
            input_dim=input_dim, d_model=64, num_heads=4, num_layers=2,
            output_dim=output_dim, forecast_horizon=forecast_horizon
        )
    else:  # mlp
        model_fn = lambda: MLPForecaster(
            input_dim=input_dim, window_size=window_size, hidden_dim=128,
            output_dim=output_dim, forecast_horizon=forecast_horizon
        )

    print(f"Using {model_type.upper()} model")

    model = fe.build(model_fn=model_fn, optimizer_fn="adam")

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        MeanSquaredError(inputs=("y_pred", "y"), outputs="mse"),
        UpdateOp(model=model, loss_name="mse")
    ])

    steps_per_epoch = len(train_X) // batch_size
    cycle_length = epochs * steps_per_epoch

    traces = [
        MAE(pred_key="y_pred", true_key="y"),
        ModelSaver(model=model, save_dir=save_dir, frequency=10),
        LRScheduler(
            model=model,
            lr_fn=lambda step: cosine_decay(
                step, cycle_length=cycle_length,
                init_lr=1e-3, min_lr=1e-5
            )
        )
    ]

    return fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--window-size", type=int, default=24,
                        help="Input window size (look-back)")
    parser.add_argument("--forecast-horizon", type=int, default=1,
                        help="Steps ahead to forecast")
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "lstm", "transformer"],
                        help="Model architecture")
    args = parser.parse_args()

    print("Time Series Forecasting with FastMLX")
    print(f"  Window size: {args.window_size}")
    print(f"  Forecast horizon: {args.forecast_horizon}")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        model_type=args.model,
    )
    est.fit()
    est.test()
