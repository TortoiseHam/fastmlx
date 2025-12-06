"""DCGAN (Deep Convolutional GAN) for image generation using :mod:`fastmlx`.

Demonstrates generative adversarial network training on MNIST.
The generator learns to create realistic digit images from random noise,
while the discriminator learns to distinguish real from generated images.

Reference:
    Radford et al., "Unsupervised Representation Learning with Deep
    Convolutional Generative Adversarial Networks", ICLR 2016.
"""

from __future__ import annotations

import argparse
import tempfile

import mlx.core as mx
import mlx.nn as nn

from fastmlx.architecture.gan import SimpleDiscriminator, SimpleGenerator
from fastmlx.dataset.data import mnist
from fastmlx.op import Op
from fastmlx.trace.base import Trace


class GeneratorOp(Op):
    """Generate fake images from random noise."""

    def __init__(
        self,
        generator: nn.Module,
        latent_dim: int,
        batch_size_key: str,
        outputs: str
    ) -> None:
        super().__init__([batch_size_key], outputs)
        self.generator = generator
        self.latent_dim = latent_dim

    def forward(self, data, state):
        x = data[0]  # Use to get batch size
        batch_size = x.shape[0]
        z = mx.random.normal((batch_size, self.latent_dim))
        fake_images = self.generator(z)
        return fake_images


class DiscriminatorLoss(Op):
    """Binary cross-entropy loss for discriminator.

    Trains discriminator to output 1 for real, 0 for fake.
    """

    def __init__(
        self,
        discriminator: nn.Module,
        inputs: tuple,
        outputs: str
    ) -> None:
        super().__init__(list(inputs), outputs)
        self.discriminator = discriminator

    def forward(self, data, state):
        real_images, fake_images = data

        # Discriminator predictions
        real_preds = self.discriminator(real_images)
        fake_preds = self.discriminator(fake_images)

        # Binary cross-entropy loss
        # Real images should be classified as 1
        real_loss = mx.mean(nn.losses.binary_cross_entropy(
            mx.sigmoid(real_preds),
            mx.ones_like(real_preds)
        ))
        # Fake images should be classified as 0
        fake_loss = mx.mean(nn.losses.binary_cross_entropy(
            mx.sigmoid(fake_preds),
            mx.zeros_like(fake_preds)
        ))

        return (real_loss + fake_loss) / 2


class GeneratorLoss(Op):
    """Binary cross-entropy loss for generator.

    Trains generator to fool discriminator (make fake images classified as 1).
    """

    def __init__(
        self,
        discriminator: nn.Module,
        inputs: str,
        outputs: str
    ) -> None:
        super().__init__([inputs], outputs)
        self.discriminator = discriminator

    def forward(self, data, state):
        fake_images = data[0]

        # Generator wants discriminator to classify fakes as real (1)
        fake_preds = self.discriminator(fake_images)
        loss = mx.mean(nn.losses.binary_cross_entropy(
            mx.sigmoid(fake_preds),
            mx.ones_like(fake_preds)
        ))

        return loss


class GANUpdateOp(Op):
    """Alternating update for GAN training.

    Updates discriminator and generator with separate optimizers.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        gen_optimizer,
        disc_optimizer,
        gen_loss_key: str,
        disc_loss_key: str,
        outputs: str = "gan_loss"
    ) -> None:
        super().__init__([gen_loss_key, disc_loss_key], outputs)
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    def forward(self, data, state):
        gen_loss, disc_loss = data

        # This op receives the already-computed losses
        # The actual gradient computation happens in the training loop
        # For now, we just return combined loss for monitoring
        return gen_loss + disc_loss


class GANMetrics(Trace):
    """Track GAN training metrics."""

    def __init__(
        self,
        gen_loss_key: str = "gen_loss",
        disc_loss_key: str = "disc_loss"
    ) -> None:
        self.gen_loss_key = gen_loss_key
        self.disc_loss_key = disc_loss_key
        self.gen_losses = []
        self.disc_losses = []

    def on_epoch_begin(self, state):
        self.gen_losses = []
        self.disc_losses = []

    def on_batch_end(self, batch, state):
        if self.gen_loss_key in batch:
            loss = batch[self.gen_loss_key]
            if isinstance(loss, mx.array):
                loss = float(loss.item())
            self.gen_losses.append(loss)
        if self.disc_loss_key in batch:
            loss = batch[self.disc_loss_key]
            if isinstance(loss, mx.array):
                loss = float(loss.item())
            self.disc_losses.append(loss)

    def on_epoch_end(self, state):
        if self.gen_losses:
            state['metrics']['gen_loss'] = sum(self.gen_losses) / len(self.gen_losses)
        if self.disc_losses:
            state['metrics']['disc_loss'] = sum(self.disc_losses) / len(self.disc_losses)


def train_gan(
    epochs: int = 50,
    batch_size: int = 64,
    latent_dim: int = 100,
    lr: float = 2e-4,
    save_dir: str = tempfile.mkdtemp(),
) -> None:
    """Train DCGAN on MNIST.

    This uses a custom training loop since GANs require alternating
    updates between generator and discriminator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        latent_dim: Dimension of latent noise vector.
        lr: Learning rate for both networks.
        save_dir: Directory to save models.
    """
    # Load data
    train_data, _ = mnist.load_data()

    # Normalize to [-1, 1] for tanh output
    images = train_data.data["x"].astype(mx.float32) / 255.0 * 2 - 1

    # Create models
    image_shape = (1, 28, 28)
    generator = SimpleGenerator(latent_dim=latent_dim, image_shape=image_shape)
    discriminator = SimpleDiscriminator(image_shape=image_shape)

    # Initialize
    mx.eval(generator.parameters())
    mx.eval(discriminator.parameters())

    # Optimizers (Adam with beta1=0.5 is standard for GANs)
    gen_optimizer = nn.optimizers.Adam(learning_rate=lr, betas=[0.5, 0.999])
    disc_optimizer = nn.optimizers.Adam(learning_rate=lr, betas=[0.5, 0.999])

    num_samples = images.shape[0]
    num_batches = num_samples // batch_size

    print("Training DCGAN on MNIST")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Generator params: {sum(p.size for p in generator.parameters().values()):,}")
    print(f"  Discriminator params: {sum(p.size for p in discriminator.parameters().values()):,}")

    def disc_loss_fn(disc_params, real_images, fake_images):
        """Discriminator loss function for gradient computation."""
        discriminator.update(disc_params)
        real_preds = discriminator(real_images)
        fake_preds = discriminator(fake_images)

        real_loss = mx.mean(nn.losses.binary_cross_entropy(
            mx.sigmoid(real_preds), mx.ones_like(real_preds)
        ))
        fake_loss = mx.mean(nn.losses.binary_cross_entropy(
            mx.sigmoid(fake_preds), mx.zeros_like(fake_preds)
        ))
        return (real_loss + fake_loss) / 2

    def gen_loss_fn(gen_params, z):
        """Generator loss function for gradient computation."""
        generator.update(gen_params)
        fake_images = generator(z)
        fake_preds = discriminator(fake_images)
        return mx.mean(nn.losses.binary_cross_entropy(
            mx.sigmoid(fake_preds), mx.ones_like(fake_preds)
        ))

    for epoch in range(epochs):
        # Shuffle data
        indices = mx.random.permutation(num_samples)
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            batch_indices = indices[start:start + batch_size]
            real_images = images[batch_indices]

            # Generate fake images
            z = mx.random.normal((batch_size, latent_dim))
            fake_images = generator(z)

            # Update discriminator
            disc_loss, disc_grads = mx.value_and_grad(disc_loss_fn)(
                discriminator.trainable_parameters(), real_images, fake_images
            )
            disc_optimizer.update(discriminator, disc_grads)

            # Update generator
            z = mx.random.normal((batch_size, latent_dim))
            gen_loss, gen_grads = mx.value_and_grad(gen_loss_fn)(
                generator.trainable_parameters(), z
            )
            gen_optimizer.update(generator, gen_grads)

            mx.eval(generator.parameters(), discriminator.parameters())

            epoch_gen_loss += float(gen_loss.item())
            epoch_disc_loss += float(disc_loss.item())

        avg_gen_loss = epoch_gen_loss / num_batches
        avg_disc_loss = epoch_disc_loss / num_batches

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")

    # Save models
    import os
    gen_path = os.path.join(save_dir, "generator.npz")
    disc_path = os.path.join(save_dir, "discriminator.npz")
    generator.save_weights(gen_path)
    discriminator.save_weights(disc_path)
    print(f"\nModels saved to {save_dir}")

    return generator, discriminator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCGAN Training with FastMLX")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent dimension")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    train_gan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
    )
