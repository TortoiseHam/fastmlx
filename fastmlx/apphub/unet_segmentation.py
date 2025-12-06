"""UNet image segmentation example using :mod:`fastmlx`.

This example demonstrates semantic segmentation using the UNet architecture
with Dice loss on synthetic circular mask data.

Reference:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation", MICCAI 2015.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Tuple

import numpy as np
import mlx.core as mx

import fastmlx as fe
from fastmlx.architecture import UNet
from fastmlx.dataset import MLXDataset
from fastmlx.op import Minmax, DiceLoss, ModelOp, UpdateOp, HorizontalFlip, VerticalFlip, Sometimes
from fastmlx.schedule import warmup_cosine_decay
from fastmlx.trace.metric import Dice
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.adapt import LRScheduler


def generate_synthetic_data(
    num_samples: int = 1000,
    image_size: int = 128,
    num_circles: int = 3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic segmentation data with circular masks.

    Creates images with random circles and corresponding binary masks.

    Args:
        num_samples: Number of samples to generate.
        image_size: Size of square images.
        num_circles: Maximum number of circles per image.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (images, masks) as numpy arrays.
    """
    np.random.seed(seed)

    images = np.zeros((num_samples, image_size, image_size, 1), dtype=np.float32)
    masks = np.zeros((num_samples, image_size, image_size, 1), dtype=np.float32)

    for i in range(num_samples):
        # Add background noise
        images[i] = np.random.uniform(0, 0.3, (image_size, image_size, 1))

        # Add random circles
        n_circles = np.random.randint(1, num_circles + 1)
        for _ in range(n_circles):
            # Random circle parameters
            cx = np.random.randint(10, image_size - 10)
            cy = np.random.randint(10, image_size - 10)
            radius = np.random.randint(5, min(25, image_size // 4))
            intensity = np.random.uniform(0.6, 1.0)

            # Create circle mask
            y, x = np.ogrid[:image_size, :image_size]
            circle_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2

            # Apply to image and mask
            images[i, circle_mask, 0] = intensity
            masks[i, circle_mask, 0] = 1.0

    return images, masks


def get_estimator(
    epochs: int = 20,
    batch_size: int = 16,
    save_dir: str = tempfile.mkdtemp(),
    image_size: int = 128,
    train_samples: int = 800,
    eval_samples: int = 200,
) -> fe.Estimator:
    """Create UNet segmentation estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        save_dir: Directory to save best model.
        image_size: Size of synthetic images.
        train_samples: Number of training samples.
        eval_samples: Number of evaluation samples.

    Returns:
        Configured Estimator ready for training.
    """
    # Generate synthetic data
    train_images, train_masks = generate_synthetic_data(
        num_samples=train_samples, image_size=image_size, seed=42
    )
    eval_images, eval_masks = generate_synthetic_data(
        num_samples=eval_samples, image_size=image_size, seed=123
    )

    train_data = MLXDataset({
        "x": mx.array(train_images),
        "y": mx.array(train_masks)
    })
    eval_data = MLXDataset({
        "x": mx.array(eval_images),
        "y": mx.array(eval_masks)
    })

    # Pipeline with augmentation
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Minmax(inputs="x", outputs="x"),
            Sometimes(HorizontalFlip(inputs="x", outputs="x")),
            Sometimes(VerticalFlip(inputs="x", outputs="x")),
            # Apply same transforms to mask
            Sometimes(HorizontalFlip(inputs="y", outputs="y")),
            Sometimes(VerticalFlip(inputs="y", outputs="y")),
        ],
    )

    # Build UNet model
    model = fe.build(
        model_fn=lambda: UNet(
            input_shape=(1, image_size, image_size),
            classes=1,  # Binary segmentation
            base_filters=32,
            depth=4
        ),
        optimizer_fn="adam"
    )

    # Network with Dice loss
    network = fe.Network([
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        DiceLoss(inputs=("y_pred", "y"), outputs="dice_loss"),
        UpdateOp(model=model, loss_name="dice_loss")
    ])

    # Calculate total steps for scheduler
    steps_per_epoch = train_samples // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = steps_per_epoch * 2  # 2 epochs warmup

    # Traces
    traces = [
        Dice(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="dice"),
        LRScheduler(
            model=model,
            lr_fn=lambda step: warmup_cosine_decay(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                init_lr=1e-3,
                min_lr=1e-5
            )
        )
    ]

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )
    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet Segmentation with FastMLX")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", type=int, default=128, help="Image size")
    args = parser.parse_args()

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    est.fit()
    est.test()
