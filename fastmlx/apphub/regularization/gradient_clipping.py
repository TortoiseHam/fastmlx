"""Gradient Clipping Techniques for Stable Training.

This example demonstrates gradient clipping techniques to prevent exploding gradients:
- Gradient norm clipping: Rescale gradients if their norm exceeds a threshold
- Gradient value clipping: Clip individual gradient values to a range

Gradient clipping is essential for:
- Training RNNs/LSTMs
- Training with large learning rates
- Handling noisy or irregular data

Example usage:
    python gradient_clipping.py --clip_norm 1.0 --epochs 10
"""

import argparse
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import fastmlx as fe
from fastmlx.dataset import MLXDataset
from fastmlx.op import ModelOp, Op
from fastmlx.trace import Trace


class GradientClipUpdateOp(Op):
    """Update operation with gradient clipping.

    Args:
        model: The model to update.
        loss_name: Name of the loss tensor.
        clip_norm: Maximum gradient norm (None to disable).
        clip_value: Maximum gradient value (None to disable).
        mode: When to execute.
    """

    def __init__(
        self,
        model: fe.build,
        loss_name: str,
        clip_norm: float | None = 1.0,
        clip_value: float | None = None,
        mode: str | list[str] = "train",
    ):
        super().__init__(inputs=loss_name, outputs=None, mode=mode)
        self.model = model
        self.clip_norm = clip_norm
        self.clip_value = clip_value

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> None:
        """Compute gradients, clip them, and update model."""
        # Loss is passed in but we get gradients from model.current_grads
        _ = data[0] if isinstance(data, list) else data

        # Get gradients
        grads = self.model.current_grads

        if grads is not None:
            # Apply gradient norm clipping
            if self.clip_norm is not None:
                grads = self._clip_by_norm(grads, self.clip_norm)

            # Apply gradient value clipping
            if self.clip_value is not None:
                grads = self._clip_by_value(grads, self.clip_value)

            # Update with clipped gradients
            self.model.optimizer.update(self.model.model, grads)

        return None

    def _clip_by_norm(self, grads: dict, max_norm: float) -> dict:
        """Clip gradients by global norm."""
        # Compute global norm
        total_norm_sq = 0.0
        for key, grad in grads.items():
            if isinstance(grad, mx.array):
                total_norm_sq += mx.sum(grad ** 2)
            elif isinstance(grad, dict):
                for v in self._flatten_grads(grad):
                    total_norm_sq += mx.sum(v ** 2)

        total_norm = mx.sqrt(total_norm_sq)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = mx.minimum(clip_coef, mx.array(1.0))

        # Scale gradients
        return self._scale_grads(grads, clip_coef)

    def _clip_by_value(self, grads: dict, max_value: float) -> dict:
        """Clip individual gradient values."""
        clipped = {}
        for key, grad in grads.items():
            if isinstance(grad, mx.array):
                clipped[key] = mx.clip(grad, -max_value, max_value)
            elif isinstance(grad, dict):
                clipped[key] = self._clip_by_value(grad, max_value)
        return clipped

    def _flatten_grads(self, grads: dict) -> list:
        """Flatten nested gradient dict to list of arrays."""
        result = []
        for v in grads.values():
            if isinstance(v, mx.array):
                result.append(v)
            elif isinstance(v, dict):
                result.extend(self._flatten_grads(v))
        return result

    def _scale_grads(self, grads: dict, scale: mx.array) -> dict:
        """Scale all gradients by a factor."""
        scaled = {}
        for key, grad in grads.items():
            if isinstance(grad, mx.array):
                scaled[key] = grad * scale
            elif isinstance(grad, dict):
                scaled[key] = self._scale_grads(grad, scale)
        return scaled


class GradientMonitor(Trace):
    """Monitor gradient statistics during training.

    Args:
        model: The model to monitor.
    """

    def __init__(self, model: fe.build):
        super().__init__()
        self.model = model
        self.grad_norms = []

    def on_batch_end(self, data: dict[str, Any]) -> None:
        """Record gradient norm after each batch."""
        if self.model.current_grads is not None:
            total_norm_sq = 0.0
            for key, grad in self.model.current_grads.items():
                if isinstance(grad, mx.array):
                    total_norm_sq += float(mx.sum(grad ** 2))

            norm = total_norm_sq ** 0.5
            self.grad_norms.append(norm)

    def on_epoch_end(self, data: dict[str, Any]) -> None:
        """Report gradient statistics for the epoch."""
        if self.grad_norms:
            avg_norm = sum(self.grad_norms) / len(self.grad_norms)
            max_norm = max(self.grad_norms)
            data["avg_grad_norm"] = avg_norm
            data["max_grad_norm"] = max_norm
            self.grad_norms = []


class SimpleRNN(nn.Module):
    """Simple RNN that can exhibit exploding gradients."""

    def __init__(self, vocab_size: int = 100, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_cell = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Embed input
        embedded = self.embedding(x.astype(mx.int32))

        # Initialize hidden state
        hidden = mx.zeros((batch_size, self.hidden_dim))

        # Process sequence
        outputs = []
        for t in range(seq_len):
            inp = embedded[:, t, :]
            combined = mx.concatenate([inp, hidden], axis=-1)
            hidden = mx.tanh(self.rnn_cell(combined))
            outputs.append(hidden)

        # Final output
        final_hidden = outputs[-1]
        return self.output(final_hidden)


def get_estimator(
    epochs: int = 10,
    batch_size: int = 64,
    seq_len: int = 50,
    clip_norm: float | None = 1.0,
    clip_value: float | None = None,
    lr: float = 0.01,
    vocab_size: int = 100,
) -> fe.Estimator:
    """Create an estimator for gradient clipping training.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        seq_len: Sequence length.
        clip_norm: Maximum gradient norm (None to disable).
        clip_value: Maximum gradient value (None to disable).
        lr: Learning rate.
        vocab_size: Vocabulary size.

    Returns:
        Configured Estimator.
    """
    import numpy as np

    # Generate synthetic sequence data
    np.random.seed(42)
    num_samples = 10000

    # Random sequences
    x_train = np.random.randint(0, vocab_size, size=(num_samples, seq_len))
    # Target is next token prediction (last token)
    y_train = np.random.randint(0, vocab_size, size=(num_samples,))

    x_test = np.random.randint(0, vocab_size, size=(num_samples // 10, seq_len))
    y_test = np.random.randint(0, vocab_size, size=(num_samples // 10,))

    # Create pipeline
    pipeline = fe.Pipeline(
        train_data=MLXDataset(data={"x": x_train, "y": y_train}),
        eval_data=MLXDataset(data={"x": x_test, "y": y_test}),
        batch_size=batch_size,
    )

    # Build model
    model = fe.build(
        model_fn=lambda: SimpleRNN(vocab_size=vocab_size),
        optimizer_fn=lambda: optim.SGD(learning_rate=lr),  # SGD can have gradient issues
    )

    # Create network with gradient clipping
    network = fe.Network(
        ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            fe.op.CrossEntropy(inputs=["y_pred", "y"], outputs="loss", mode="train"),
            GradientClipUpdateOp(
                model=model,
                loss_name="loss",
                clip_norm=clip_norm,
                clip_value=clip_value,
            ),
        ]
    )

    # Create estimator
    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=[
            GradientMonitor(model=model),
        ],
        log_interval=50,
    )

    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Clipping Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--clip_value", type=float, default=None, help="Max gradient value")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        clip_norm=args.clip_norm,
        clip_value=args.clip_value,
        lr=args.lr,
    )
    est.fit()
    est.test()
