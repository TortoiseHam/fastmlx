"""Conditional GAN for Class-Conditional Image Generation.

This example demonstrates Conditional GAN (cGAN), which generates images
conditioned on class labels.

Unlike regular GANs that generate random samples, cGANs can generate:
- Specific digits (e.g., "generate a 7")
- Specific categories (e.g., "generate a cat")

Architecture:
- Generator: Takes noise + class label -> Image
- Discriminator: Takes image + class label -> Real/Fake

Example usage:
    python conditional_gan.py --epochs 50 --latent_dim 100
"""

import argparse
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import fastmlx as fe
from fastmlx.dataset import MLXDataset
from fastmlx.op import Op
from fastmlx.trace import Trace


class ConditionalGenerator(nn.Module):
    """Generator conditioned on class labels.

    Args:
        latent_dim: Dimension of noise vector.
        num_classes: Number of class labels.
        img_shape: Output image shape (H, W, C).
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        img_shape: tuple[int, int, int] = (28, 28, 1),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_shape = img_shape

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 50)

        # Generator layers
        input_dim = latent_dim + 50  # noise + label embedding
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm(512)
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm(1024)
        self.fc4 = nn.Linear(1024, int(np.prod(img_shape)))

    def __call__(self, noise: mx.array, labels: mx.array) -> mx.array:
        # Embed labels
        label_emb = self.label_embedding(labels.astype(mx.int32))

        # Concatenate noise and label embedding
        x = mx.concatenate([noise, label_emb], axis=1)

        # Generate
        x = nn.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        x = nn.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        x = nn.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.2)
        x = mx.tanh(self.fc4(x))

        # Reshape to image
        x = x.reshape(-1, *self.img_shape)

        return x


class ConditionalDiscriminator(nn.Module):
    """Discriminator conditioned on class labels.

    Args:
        num_classes: Number of class labels.
        img_shape: Input image shape (H, W, C).
    """

    def __init__(
        self,
        num_classes: int = 10,
        img_shape: tuple[int, int, int] = (28, 28, 1),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape

        # Label embedding - map to image-sized embedding
        self.label_embedding = nn.Embedding(num_classes, int(np.prod(img_shape)))

        # Discriminator layers (input: flattened image + label embedding)
        input_dim = int(np.prod(img_shape)) * 2
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)

    def __call__(self, img: mx.array, labels: mx.array) -> mx.array:
        # Flatten image
        img_flat = img.reshape(img.shape[0], -1)

        # Embed labels
        label_emb = self.label_embedding(labels.astype(mx.int32))

        # Concatenate
        x = mx.concatenate([img_flat, label_emb], axis=1)

        # Discriminate
        x = nn.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.dropout(x)
        x = nn.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class GenerateNoiseOp(Op):
    """Generate random noise for generator."""

    def __init__(
        self,
        outputs: str,
        latent_dim: int = 100,
        mode: str | list[str] | None = None,
    ):
        super().__init__(inputs=None, outputs=outputs, mode=mode)
        self.latent_dim = latent_dim

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> mx.array:
        """Generate random noise."""
        batch_size = state.get("batch_size", 64)
        return mx.random.normal(shape=(batch_size, self.latent_dim))


class GANLossOp(Op):
    """Binary cross entropy loss for GAN training."""

    def __init__(
        self,
        inputs: str,
        outputs: str,
        real: bool = True,
        mode: str | list[str] | None = None,
    ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.real = real

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> mx.array:
        """Compute BCE loss."""
        logits = data[0] if isinstance(data, list) else data

        if self.real:
            # Real samples should be classified as 1
            labels = mx.ones(logits.shape)
        else:
            # Fake samples should be classified as 0
            labels = mx.zeros(logits.shape)

        # BCE with logits
        loss = mx.mean(nn.losses.binary_cross_entropy(
            mx.sigmoid(logits), labels, reduction="none"
        ))

        return loss


class cGANTrainingOp(Op):
    """Training operation for conditional GAN.

    Handles alternating generator/discriminator updates.
    """

    def __init__(
        self,
        generator: fe.build,
        discriminator: fe.build,
        latent_dim: int = 100,
        mode: str | list[str] = "train",
    ):
        super().__init__(inputs="x", outputs=["g_loss", "d_loss", "fake_imgs"], mode=mode)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def forward(self, data: list[mx.array], state: dict[str, Any]) -> tuple[mx.array, mx.array, mx.array]:
        """Perform one training step for both G and D."""
        real_imgs = data[0] if isinstance(data, list) else data
        labels = state.get("y", mx.zeros(real_imgs.shape[0], dtype=mx.int32))
        batch_size = real_imgs.shape[0]

        # Ensure correct shape
        if real_imgs.ndim == 3:
            real_imgs = mx.expand_dims(real_imgs, axis=-1)

        # Normalize to [-1, 1] for tanh output
        real_imgs = real_imgs * 2 - 1

        # Generate noise
        noise = mx.random.normal(shape=(batch_size, self.latent_dim))

        # Train Discriminator
        def d_loss_fn(d_params):
            self.discriminator.model.update(d_params)

            # Real loss
            real_logits = self.discriminator.model(real_imgs, labels)
            real_loss = mx.mean(nn.losses.binary_cross_entropy(
                mx.sigmoid(real_logits),
                mx.ones(real_logits.shape),
                reduction="none"
            ))

            # Fake loss
            fake_imgs = self.generator.model(noise, labels)
            fake_logits = self.discriminator.model(fake_imgs, labels)
            fake_loss = mx.mean(nn.losses.binary_cross_entropy(
                mx.sigmoid(fake_logits),
                mx.zeros(fake_logits.shape),
                reduction="none"
            ))

            return (real_loss + fake_loss) / 2

        d_loss, d_grads = mx.value_and_grad(d_loss_fn)(self.discriminator.model.parameters())
        self.discriminator.optimizer.update(self.discriminator.model, d_grads)
        mx.eval(self.discriminator.model.parameters())

        # Train Generator
        def g_loss_fn(g_params):
            self.generator.model.update(g_params)

            fake_imgs = self.generator.model(noise, labels)
            fake_logits = self.discriminator.model(fake_imgs, labels)

            # Generator wants discriminator to think fakes are real
            g_loss = mx.mean(nn.losses.binary_cross_entropy(
                mx.sigmoid(fake_logits),
                mx.ones(fake_logits.shape),
                reduction="none"
            ))

            return g_loss

        g_loss, g_grads = mx.value_and_grad(g_loss_fn)(self.generator.model.parameters())
        self.generator.optimizer.update(self.generator.model, g_grads)
        mx.eval(self.generator.model.parameters())

        # Generate samples for visualization
        fake_imgs = self.generator.model(noise, labels)

        return g_loss, d_loss, fake_imgs


class GANMonitor(Trace):
    """Monitor GAN training progress."""

    def __init__(self, generator: fe.build, latent_dim: int = 100, num_classes: int = 10):
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def on_epoch_end(self, data: dict[str, Any]) -> None:
        """Generate samples for each class."""
        # Generate one sample per class
        noise = mx.random.normal(shape=(self.num_classes, self.latent_dim))
        labels = mx.arange(self.num_classes)

        samples = self.generator.model(noise, labels)
        mx.eval(samples)

        # Report generation is working
        data["samples_generated"] = True


def get_estimator(
    epochs: int = 50,
    batch_size: int = 128,
    latent_dim: int = 100,
    lr: float = 0.0002,
) -> fe.Estimator:
    """Create an estimator for conditional GAN training.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        latent_dim: Dimension of latent noise vector.
        lr: Learning rate for both G and D.

    Returns:
        Configured Estimator.
    """
    from fastmlx.dataset.data import mnist

    # Load MNIST dataset
    train_data, test_data = mnist.load_data()

    # Create pipeline
    pipeline = fe.Pipeline(
        train_data=MLXDataset(data={"x": train_data[0], "y": train_data[1]}),
        batch_size=batch_size,
        ops=[
            fe.op.Normalize(inputs="x", outputs="x", mean=0.0, std=1.0),
        ],
    )

    # Build generator and discriminator
    generator = fe.build(
        model_fn=lambda: ConditionalGenerator(latent_dim=latent_dim),
        optimizer_fn=lambda: optim.Adam(learning_rate=lr, betas=[0.5, 0.999]),
    )

    discriminator = fe.build(
        model_fn=lambda: ConditionalDiscriminator(),
        optimizer_fn=lambda: optim.Adam(learning_rate=lr, betas=[0.5, 0.999]),
    )

    # Create network with custom training op
    network = fe.Network(
        ops=[
            cGANTrainingOp(
                generator=generator,
                discriminator=discriminator,
                latent_dim=latent_dim,
            ),
        ]
    )

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=[
            GANMonitor(generator=generator, latent_dim=latent_dim),
        ],
        log_interval=100,
    )

    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent dimension")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    args = parser.parse_args()

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
    )
    est.fit()
