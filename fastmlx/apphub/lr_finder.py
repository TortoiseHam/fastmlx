"""Learning Rate Finder example using :mod:`fastmlx`.

Implements the learning rate range test to help find optimal learning rates.
The idea is to train with exponentially increasing learning rates and plot
loss vs learning rate to find the sweet spot.

Good learning rates are typically:
- Just before the loss starts increasing rapidly
- Where the loss decreases fastest (steepest negative slope)

Reference:
    Smith, "Cyclical Learning Rates for Training Neural Networks", WACV 2017.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from fastmlx.architecture import ResNet9
from fastmlx.dataset.data import cifair10


def lr_finder(
    model: nn.Module,
    train_data,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iterations: int = 100,
    batch_size: int = 64,
    smooth_factor: float = 0.05,
) -> Tuple[List[float], List[float]]:
    """Find optimal learning rate using the LR range test.

    Trains the model with exponentially increasing learning rates and
    records the loss at each step.

    Args:
        model: The model to train.
        train_data: Training dataset.
        start_lr: Starting learning rate (very small).
        end_lr: Ending learning rate (very large).
        num_iterations: Number of iterations to run.
        batch_size: Batch size for training.
        smooth_factor: Smoothing factor for loss (exponential moving average).

    Returns:
        Tuple of (learning_rates, losses) lists.
    """
    # Calculate LR multiplier
    lr_mult = (end_lr / start_lr) ** (1 / num_iterations)

    # Prepare data
    images = train_data.data["x"].astype(mx.float32) / 255.0
    labels = train_data.data["y"]

    # Normalize images
    mean = mx.array([0.4914, 0.4822, 0.4465])
    std = mx.array([0.2471, 0.2435, 0.2616])
    images = (images - mean) / std

    num_samples = images.shape[0]

    # Initialize optimizer with starting LR
    optimizer = nn.optimizers.SGD(learning_rate=start_lr, momentum=0.9)

    learning_rates = []
    losses = []
    best_loss = float('inf')
    avg_loss = 0.0
    current_lr = start_lr

    def loss_fn(params, x, y):
        model.update(params)
        logits = model(x)
        # Cross-entropy loss
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        loss = -mx.mean(mx.take_along_axis(
            log_probs, y[:, None].astype(mx.int32), axis=1
        ))
        return loss

    print(f"Running LR Finder: {start_lr:.2e} -> {end_lr:.2e}")
    print("-" * 50)

    for iteration in range(num_iterations):
        # Get random batch
        indices = mx.random.randint(0, num_samples, (batch_size,))
        batch_x = images[indices]
        batch_y = labels[indices]

        # Forward and backward
        loss, grads = mx.value_and_grad(loss_fn)(
            model.trainable_parameters(), batch_x, batch_y
        )

        # Update model
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # Smooth loss
        loss_val = float(loss.item())
        if iteration == 0:
            avg_loss = loss_val
        else:
            avg_loss = smooth_factor * loss_val + (1 - smooth_factor) * avg_loss

        # Record
        learning_rates.append(current_lr)
        losses.append(avg_loss)

        # Check for divergence
        if loss_val > 4 * best_loss or math.isnan(loss_val):
            print(f"Stopping early at LR={current_lr:.2e} (loss diverged)")
            break

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Progress
        if (iteration + 1) % 10 == 0:
            print(f"  Iter {iteration + 1}/{num_iterations}: "
                  f"LR={current_lr:.2e}, Loss={avg_loss:.4f}")

        # Increase learning rate
        current_lr *= lr_mult
        optimizer.learning_rate = current_lr

    return learning_rates, losses


def suggest_lr(learning_rates: List[float], losses: List[float]) -> float:
    """Suggest optimal learning rate from LR finder results.

    Finds the learning rate where loss is decreasing fastest.

    Args:
        learning_rates: List of learning rates tested.
        losses: Corresponding losses.

    Returns:
        Suggested learning rate.
    """
    if len(losses) < 3:
        return learning_rates[0]

    # Find steepest negative slope
    min_grad = 0
    best_idx = 0

    for i in range(1, len(losses) - 1):
        # Gradient (in log space for LR)
        grad = (losses[i + 1] - losses[i - 1]) / 2

        if grad < min_grad:
            min_grad = grad
            best_idx = i

    # Return LR slightly before steepest descent
    suggested_idx = max(0, best_idx - 2)
    return learning_rates[suggested_idx]


def plot_lr_finder(
    learning_rates: List[float],
    losses: List[float],
    suggested_lr: float = None,
    save_path: str = None
) -> None:
    """Plot learning rate finder results.

    Args:
        learning_rates: List of learning rates.
        losses: Corresponding losses.
        suggested_lr: Optionally mark suggested LR.
        save_path: Path to save plot (requires matplotlib).
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(learning_rates, losses)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        ax.grid(True)

        if suggested_lr:
            ax.axvline(x=suggested_lr, color='r', linestyle='--',
                      label=f'Suggested LR: {suggested_lr:.2e}')
            ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    except ImportError:
        print("\nmatplotlib not installed. Printing results:")
        print(f"{'LR':>12} | {'Loss':>10}")
        print("-" * 25)
        for lr, loss in zip(learning_rates[::5], losses[::5]):
            print(f"{lr:>12.2e} | {loss:>10.4f}")


def run_lr_finder(
    batch_size: int = 64,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iterations: int = 100,
    plot: bool = True,
) -> float:
    """Run learning rate finder on CIFAR-10.

    Args:
        batch_size: Batch size.
        start_lr: Starting learning rate.
        end_lr: Ending learning rate.
        num_iterations: Number of iterations.
        plot: Whether to plot results.

    Returns:
        Suggested learning rate.
    """
    # Load data
    train_data, _ = cifair10.load_data()

    # Create fresh model
    model = ResNet9(input_shape=(3, 32, 32))
    mx.eval(model.parameters())

    print(f"Model parameters: {sum(p.size for p in model.parameters().values()):,}")

    # Run LR finder
    lrs, losses = lr_finder(
        model=model,
        train_data=train_data,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iterations=num_iterations,
        batch_size=batch_size,
    )

    # Suggest LR
    suggested = suggest_lr(lrs, losses)
    print(f"\nSuggested learning rate: {suggested:.2e}")
    print("(Use a value slightly lower for safety)")

    if plot:
        plot_lr_finder(lrs, losses, suggested_lr=suggested)

    return suggested


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Rate Finder with FastMLX")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--start-lr", type=float, default=1e-7, help="Starting LR")
    parser.add_argument("--end-lr", type=float, default=10.0, help="Ending LR")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    run_lr_finder(
        batch_size=args.batch_size,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iterations=args.iterations,
        plot=not args.no_plot,
    )
