"""Autoencoder and VAE example on MNIST using :mod:`fastmlx`.

Demonstrates unsupervised representation learning with:
1. Standard Autoencoder - learns deterministic latent representations
2. Variational Autoencoder (VAE) - learns probabilistic latent space

The VAE can generate new samples by sampling from the learned distribution.

Reference:
    Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Any, MutableMapping, Sequence

import mlx.core as mx

import fastmlx as fe
from fastmlx.architecture import Autoencoder, VAE
from fastmlx.dataset.data import mnist
from fastmlx.op import Minmax, MeanSquaredError, ModelOp, UpdateOp, Op
from fastmlx.schedule import cosine_decay
from fastmlx.trace.base import Trace
from fastmlx.trace.io import ModelSaver
from fastmlx.trace.adapt import LRScheduler


class VAEModelOp(Op):
    """Model operation for VAE that returns reconstruction, mu, and log_var."""

    def __init__(self, model, inputs: str, outputs: Sequence[str]) -> None:
        super().__init__([inputs], list(outputs))
        self.model = model

    def forward(self, data, state):
        x = data[0]
        x_recon, mu, log_var = self.model(x)
        return x_recon, mu, log_var


class VAELoss(Op):
    """Combined reconstruction + KL divergence loss for VAE.

    Loss = MSE(x, x_recon) + beta * KL(q(z|x) || p(z))

    The KL term encourages the latent distribution to match a standard normal.
    """

    def __init__(
        self,
        inputs: Sequence[str],
        outputs: str,
        beta: float = 1.0
    ) -> None:
        """Initialize VAE loss.

        Args:
            inputs: Tuple of (x, x_recon, mu, log_var) keys.
            outputs: Output key for total loss.
            beta: Weight for KL divergence term (beta-VAE).
        """
        super().__init__(list(inputs), outputs)
        self.beta = beta

    def forward(
        self,
        data: Sequence[mx.array],
        state: MutableMapping[str, Any]
    ) -> mx.array:
        x, x_recon, mu, log_var = data

        # Reconstruction loss (MSE)
        recon_loss = mx.mean((x - x_recon) ** 2)

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * mx.mean(1 + log_var - mu ** 2 - mx.exp(log_var))

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss


class ReconstructionLoss(Trace):
    """Track reconstruction loss over an epoch."""

    def __init__(self, x_key: str = "x", recon_key: str = "x_recon") -> None:
        self.x_key = x_key
        self.recon_key = recon_key
        self.total_loss = 0.0
        self.count = 0

    def on_epoch_begin(self, state):
        self.total_loss = 0.0
        self.count = 0

    def on_batch_end(self, batch, state):
        x = batch[self.x_key]
        x_recon = batch[self.recon_key]
        mse = float(mx.mean((x - x_recon) ** 2).item())
        self.total_loss += mse
        self.count += 1

    def on_epoch_end(self, state):
        state['metrics']['recon_loss'] = self.total_loss / max(1, self.count)


def get_autoencoder_estimator(
    epochs: int = 20,
    batch_size: int = 64,
    latent_dim: int = 32,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create standard Autoencoder estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        latent_dim: Dimension of latent space.
        save_dir: Directory to save models.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = mnist.load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[Minmax(inputs="x", outputs="x")],
    )

    model = fe.build(
        model_fn=lambda: Autoencoder(
            input_shape=(1, 28, 28),
            latent_dim=latent_dim
        ),
        optimizer_fn="adam"
    )

    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="x_recon"),
        MeanSquaredError(inputs=("x_recon", "x"), outputs="mse_loss"),
        UpdateOp(model=model, loss_name="mse_loss")
    ])

    steps_per_epoch = 60000 // batch_size
    cycle_length = epochs * steps_per_epoch

    traces = [
        ReconstructionLoss(x_key="x", recon_key="x_recon"),
        ModelSaver(model=model, save_dir=save_dir, frequency=5),
        LRScheduler(
            model=model,
            lr_fn=lambda step: cosine_decay(
                step,
                cycle_length=cycle_length,
                init_lr=1e-3,
                min_lr=1e-5
            )
        )
    ]

    return fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )


def get_vae_estimator(
    epochs: int = 20,
    batch_size: int = 64,
    latent_dim: int = 32,
    beta: float = 1.0,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create Variational Autoencoder estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        latent_dim: Dimension of latent space.
        beta: Weight for KL divergence (beta-VAE).
        save_dir: Directory to save models.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = mnist.load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[Minmax(inputs="x", outputs="x")],
    )

    model = fe.build(
        model_fn=lambda: VAE(
            input_shape=(1, 28, 28),
            latent_dim=latent_dim
        ),
        optimizer_fn="adam"
    )

    network = fe.Network([
        VAEModelOp(model=model, inputs="x", outputs=("x_recon", "mu", "log_var")),
        VAELoss(
            inputs=("x", "x_recon", "mu", "log_var"),
            outputs="vae_loss",
            beta=beta
        ),
        UpdateOp(model=model, loss_name="vae_loss")
    ])

    steps_per_epoch = 60000 // batch_size
    cycle_length = epochs * steps_per_epoch

    traces = [
        ReconstructionLoss(x_key="x", recon_key="x_recon"),
        ModelSaver(model=model, save_dir=save_dir, frequency=5),
        LRScheduler(
            model=model,
            lr_fn=lambda step: cosine_decay(
                step,
                cycle_length=cycle_length,
                init_lr=1e-3,
                min_lr=1e-5
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
    parser = argparse.ArgumentParser(
        description="Autoencoder/VAE training with FastMLX"
    )
    parser.add_argument("--model", type=str, choices=["ae", "vae"], default="vae",
                        help="Model type: 'ae' for autoencoder, 'vae' for VAE")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="Latent space dimension")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Beta for VAE (KL weight)")
    args = parser.parse_args()

    if args.model == "vae":
        print(f"Training Variational Autoencoder (beta={args.beta})")
        est = get_vae_estimator(
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            beta=args.beta,
        )
    else:
        print("Training Standard Autoencoder")
        est = get_autoencoder_estimator(
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
        )

    est.fit()
    est.test()
