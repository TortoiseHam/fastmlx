"""Gradient Accumulation example using :mod:`fastmlx`.

Demonstrates training with effective batch sizes larger than GPU memory
allows by accumulating gradients over multiple mini-batches.

This is essential for:
- Training large models on limited hardware
- Reproducing results from papers with large batch sizes
- Stable training with batch normalization

Note: With gradient accumulation, the effective batch size is:
    effective_batch = mini_batch_size * accumulation_steps
"""

from __future__ import annotations

import argparse
import tempfile

import mlx.core as mx
import mlx.nn as nn

from fastmlx.architecture import ResNet9
from fastmlx.dataset.data import cifair10
from fastmlx.op import Op
from fastmlx.schedule import warmup_cosine_decay


class GradientAccumulationUpdateOp(Op):
    """Update model weights with gradient accumulation.

    Accumulates gradients over multiple steps before applying update.

    Args:
        model: The model to update.
        loss_name: Key for the loss value in batch.
        accumulation_steps: Number of steps to accumulate before update.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        loss_name: str,
        accumulation_steps: int = 4
    ) -> None:
        super().__init__([loss_name], loss_name)
        self.model = model
        self.optimizer = optimizer
        self.loss_name = loss_name
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads = None
        self.step_count = 0

    def forward(self, data, state):
        # The loss is already computed; we need to trigger backward pass
        # This is a simplified version - full implementation would integrate
        # with the training loop

        # data is a single array when there's one input
        loss = data if not isinstance(data, list) else data[0]

        # In practice, gradient accumulation requires integration with
        # the training loop. This example shows the concept.
        self.step_count += 1

        if self.step_count >= self.accumulation_steps:
            # Apply update
            self.step_count = 0

        return loss


def train_with_accumulation(
    epochs: int = 100,
    mini_batch_size: int = 32,
    accumulation_steps: int = 4,
    target_lr: float = 1e-3,
    save_dir: str = tempfile.mkdtemp(),
) -> None:
    """Train with gradient accumulation using custom training loop.

    Args:
        epochs: Number of training epochs.
        mini_batch_size: Actual batch size that fits in memory.
        accumulation_steps: Number of steps to accumulate gradients.
        target_lr: Learning rate (scaled for effective batch size).
        save_dir: Directory to save models.
    """
    effective_batch_size = mini_batch_size * accumulation_steps
    print("Training with Gradient Accumulation")
    print(f"  Mini-batch size: {mini_batch_size}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Learning rate: {target_lr}")

    # Load data
    train_data, eval_data = cifair10.load_data()

    # Prepare data
    train_x = train_data.data["x"].astype(mx.float32) / 255.0
    train_y = train_data.data["y"]
    eval_x = eval_data.data["x"].astype(mx.float32) / 255.0
    eval_y = eval_data.data["y"]

    # Normalize
    mean = mx.array([0.4914, 0.4822, 0.4465])
    std = mx.array([0.2471, 0.2435, 0.2616])
    train_x = (train_x - mean) / std
    eval_x = (eval_x - mean) / std

    num_samples = train_x.shape[0]
    num_eval = eval_x.shape[0]

    # Create model
    model = ResNet9(input_shape=(3, 32, 32))
    mx.eval(model.parameters())

    # Optimizer
    optimizer = nn.optimizers.Adam(learning_rate=target_lr)

    steps_per_epoch = num_samples // mini_batch_size // accumulation_steps
    total_steps = epochs * steps_per_epoch

    print(f"  Steps per epoch: {steps_per_epoch}")

    def loss_fn(params, x, y):
        model.update(params)
        logits = model(x)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        loss = -mx.mean(mx.take_along_axis(
            log_probs, y[:, None].astype(mx.int32), axis=1
        ))
        return loss

    loss_and_grad = mx.value_and_grad(loss_fn)

    for epoch in range(epochs):
        # Shuffle data
        indices = mx.random.permutation(num_samples)
        train_x_shuffled = train_x[indices]
        train_y_shuffled = train_y[indices]

        epoch_loss = 0.0
        num_updates = 0

        # Initialize accumulated gradients
        accumulated_grads = None

        for step in range(num_samples // mini_batch_size):
            start = step * mini_batch_size
            batch_x = train_x_shuffled[start:start + mini_batch_size]
            batch_y = train_y_shuffled[start:start + mini_batch_size]

            # Compute loss and gradients
            loss, grads = loss_and_grad(model.trainable_parameters(), batch_x, batch_y)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = {
                    k: accumulated_grads[k] + grads[k]
                    for k in grads
                }

            # Update every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                # Average the accumulated gradients
                averaged_grads = {
                    k: v / accumulation_steps
                    for k, v in accumulated_grads.items()
                }

                # Apply update
                optimizer.update(model, averaged_grads)
                mx.eval(model.parameters())

                epoch_loss += float(loss.item())
                num_updates += 1

                # Reset accumulated gradients
                accumulated_grads = None

        avg_loss = epoch_loss / max(1, num_updates)

        # Evaluate
        eval_correct = 0
        for i in range(0, num_eval, mini_batch_size):
            batch_x = eval_x[i:i + mini_batch_size]
            batch_y = eval_y[i:i + mini_batch_size]
            logits = model(batch_x)
            preds = mx.argmax(logits, axis=-1)
            eval_correct += int(mx.sum(preds == batch_y).item())

        eval_acc = eval_correct / num_eval

        # Update learning rate (cosine decay)
        current_step = epoch * steps_per_epoch
        lr = warmup_cosine_decay(
            current_step,
            warmup_steps=steps_per_epoch * 5,
            total_steps=total_steps,
            init_lr=target_lr,
            min_lr=target_lr * 0.01
        )
        optimizer.learning_rate = lr

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Loss: {avg_loss:.4f}, Eval Acc: {eval_acc:.4f}, LR: {lr:.6f}")

    # Save model
    import os
    model_path = os.path.join(save_dir, "model.npz")
    model.save_weights(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gradient Accumulation Training with FastMLX"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--mini-batch-size", type=int, default=32,
                        help="Mini-batch size (fits in memory)")
    parser.add_argument("--accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_with_accumulation(
        epochs=args.epochs,
        mini_batch_size=args.mini_batch_size,
        accumulation_steps=args.accumulation_steps,
        target_lr=args.lr,
    )
