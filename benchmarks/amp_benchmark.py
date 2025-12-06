#!/usr/bin/env python3
"""Benchmark comparing float32 vs float16 training performance on MLX.

This script measures:
1. Forward pass time
2. Backward pass time
3. Full training step time
4. Memory usage
5. Throughput (samples/sec)

Run with: python benchmarks/amp_benchmark.py
"""

import time
from typing import Dict, Any, Tuple
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# ============================================================================
# Models for benchmarking
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for baseline benchmarks."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvNet(nn.Module):
    """CNN for image benchmarks."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # MLX Conv2d expects (N, H, W, C) format
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def __call__(self, x):
        # x: (N, H, W, C) - MLX native format, no transpose needed
        x = nn.relu(self.conv1(x))
        x = nn.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.relu(self.conv2(x))
        x = nn.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.relu(self.conv3(x))
        x = nn.max_pool2d(x, kernel_size=2, stride=2)

        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block for sequence benchmarks."""

    def __init__(self, dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SmallTransformer(nn.Module):
    """Small transformer model for benchmarking."""

    def __init__(self, vocab_size: int = 10000, dim: int = 256, num_heads: int = 4,
                 num_layers: int = 4, max_seq_len: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        self.blocks = [TransformerBlock(dim, num_heads) for _ in range(num_layers)]
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = mx.arange(seq_len)

        x = self.embedding(x) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)
        return x


# ============================================================================
# Benchmark utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count model parameters."""
    total = 0
    params = model.parameters()

    def count_dict(d):
        count = 0
        for v in d.values():
            if isinstance(v, mx.array):
                count += v.size
            elif isinstance(v, dict):
                count += count_dict(v)
        return count

    return count_dict(params)


def cast_model(model: nn.Module, dtype: mx.Dtype) -> nn.Module:
    """Cast model parameters to dtype."""
    params = model.parameters()

    def cast_dict(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, mx.array):
                result[k] = v.astype(dtype)
            elif isinstance(v, dict):
                result[k] = cast_dict(v)
            else:
                result[k] = v
        return result

    model.update(cast_dict(params))
    return model


def benchmark_forward(
    model: nn.Module,
    x: mx.array,
    num_iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """Benchmark forward pass."""
    # Warmup
    for _ in range(warmup):
        y = model(x)
        mx.eval(y)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        y = model(x)
        mx.eval(y)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "throughput": x.shape[0] / (sum(times) / len(times)),
    }


def benchmark_training_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    x: mx.array,
    y: mx.array,
    num_iterations: int = 50,
    warmup: int = 5
) -> Dict[str, float]:
    """Benchmark full training step (forward + backward + update)."""

    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup
    for _ in range(warmup):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    # Benchmark
    times = []
    losses = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        end = time.perf_counter()
        times.append(end - start)
        losses.append(float(loss))

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "throughput": x.shape[0] / (sum(times) / len(times)),
        "final_loss": losses[-1],
    }


def run_benchmark(
    name: str,
    model_fn,
    input_fn,
    label_fn,
    batch_sizes: list = [32, 64, 128],
    num_iterations: int = 50
) -> Dict[str, Any]:
    """Run complete benchmark comparing float32 vs float16."""

    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")

    results = {"name": name, "batch_sizes": {}}

    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")
        print(f"  {'-'*40}")

        batch_results = {}

        for dtype_name, dtype in [("float32", mx.float32), ("float16", mx.float16)]:
            # Create fresh model and optimizer
            model = model_fn()
            model = cast_model(model, dtype)
            optimizer = optim.Adam(learning_rate=1e-3)

            # Create data in appropriate dtype
            x = input_fn(batch_size).astype(dtype)
            y = label_fn(batch_size)

            # Count parameters
            num_params = count_parameters(model)
            memory_mb = (num_params * (2 if dtype == mx.float16 else 4)) / (1024 * 1024)

            # Run benchmark
            step_results = benchmark_training_step(
                model, optimizer, x, y,
                num_iterations=num_iterations,
                warmup=10
            )

            batch_results[dtype_name] = {
                "params": num_params,
                "memory_mb": memory_mb,
                **step_results
            }

            print(f"    {dtype_name}: {step_results['mean_ms']:.2f} ms/step, "
                  f"{step_results['throughput']:.1f} samples/sec, "
                  f"{memory_mb:.1f} MB")

        # Calculate speedup
        speedup = batch_results["float32"]["mean_ms"] / batch_results["float16"]["mean_ms"]
        memory_savings = batch_results["float32"]["memory_mb"] - batch_results["float16"]["memory_mb"]

        batch_results["speedup"] = speedup
        batch_results["memory_savings_mb"] = memory_savings

        print(f"    Speedup: {speedup:.2f}x")
        print(f"    Memory savings: {memory_savings:.1f} MB ({50:.0f}%)")

        results["batch_sizes"][batch_size] = batch_results

    return results


# ============================================================================
# Main benchmark suite
# ============================================================================

def main():
    print("=" * 60)
    print("FastMLX AMP Benchmark")
    print("Comparing float32 vs float16 training performance")
    print("=" * 60)

    # Check device
    try:
        device = mx.default_device()
        print(f"\nDevice: {device}")
    except:
        print("\nDevice: Unknown")

    all_results = []

    # Benchmark 1: Simple MLP
    print("\n" + "=" * 60)
    print("Test 1: Simple MLP (784 -> 512 -> 512 -> 10)")
    results = run_benchmark(
        name="SimpleMLP",
        model_fn=lambda: SimpleMLP(784, 512, 10),
        input_fn=lambda bs: mx.random.normal((bs, 784)),
        label_fn=lambda bs: mx.random.randint(0, 10, (bs,)),
        batch_sizes=[64, 128, 256, 512],
        num_iterations=100
    )
    all_results.append(results)

    # Benchmark 2: CNN
    print("\n" + "=" * 60)
    print("Test 2: ConvNet (32x32 images)")
    results = run_benchmark(
        name="ConvNet",
        model_fn=lambda: ConvNet(10),
        input_fn=lambda bs: mx.random.normal((bs, 32, 32, 3)),
        label_fn=lambda bs: mx.random.randint(0, 10, (bs,)),
        batch_sizes=[32, 64, 128],
        num_iterations=50
    )
    all_results.append(results)

    # Benchmark 3: Small Transformer
    print("\n" + "=" * 60)
    print("Test 3: Small Transformer (4 layers, dim=256)")
    results = run_benchmark(
        name="SmallTransformer",
        model_fn=lambda: SmallTransformer(vocab_size=10000, dim=256, num_heads=4, num_layers=4),
        input_fn=lambda bs: mx.random.randint(0, 10000, (bs, 64)),
        label_fn=lambda bs: mx.random.randint(0, 10000, (bs, 64)),
        batch_sizes=[16, 32, 64],
        num_iterations=30
    )
    all_results.append(results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for result in all_results:
        print(f"\n{result['name']}:")
        for bs, data in result["batch_sizes"].items():
            print(f"  Batch {bs}: {data['speedup']:.2f}x speedup, "
                  f"{data['memory_savings_mb']:.1f} MB saved")

    # Overall assessment
    avg_speedups = []
    for result in all_results:
        for bs, data in result["batch_sizes"].items():
            avg_speedups.append(data["speedup"])

    avg_speedup = sum(avg_speedups) / len(avg_speedups)

    print(f"\n{'='*60}")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Memory savings: 50% (by design - half precision)")
    print(f"{'='*60}")

    if avg_speedup > 1.5:
        print("\nConclusion: float16 provides significant speedup on this hardware!")
    elif avg_speedup > 1.1:
        print("\nConclusion: float16 provides modest speedup. Main benefit is memory savings.")
    else:
        print("\nConclusion: float16 provides minimal speedup. Use mainly for memory savings.")


if __name__ == "__main__":
    main()
