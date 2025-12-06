"""GPT Language Model training example using :mod:`fastmlx`.

Demonstrates training a small GPT-style decoder-only transformer
for character-level language modeling on Shakespeare text.

Reference:
    Radford et al., "Language Models are Unsupervised Multitask Learners", 2019.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request

import mlx.core as mx

import fastmlx as fe
from fastmlx.architecture import GPT
from fastmlx.dataset import MLXDataset
from fastmlx.op import Op
from fastmlx.schedule import warmup_cosine_decay
from fastmlx.trace.adapt import LRScheduler
from fastmlx.trace.base import Trace
from fastmlx.trace.io import ModelSaver

# Shakespeare dataset URL
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare(cache_dir: str = None) -> str:
    """Download Shakespeare text dataset."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), "fastmlx_data")
    os.makedirs(cache_dir, exist_ok=True)

    filepath = os.path.join(cache_dir, "shakespeare.txt")
    if not os.path.exists(filepath):
        print("Downloading Shakespeare dataset...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, filepath)
        print(f"Downloaded to {filepath}")

    with open(filepath, "r") as f:
        text = f.read()
    return text


class CharacterTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list:
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices: list) -> str:
        return "".join([self.idx_to_char[i] for i in indices])


class LanguageModelLoss(Op):
    """Cross-entropy loss for next-token prediction."""

    def __init__(self, inputs: tuple, outputs: str, vocab_size: int) -> None:
        super().__init__(list(inputs), outputs)
        self.vocab_size = vocab_size

    def forward(self, data, state):
        logits, targets = data
        # logits: (batch, seq_len, vocab_size)
        # targets: (batch, seq_len)

        # Flatten for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1).astype(mx.int32)

        # Cross-entropy loss
        log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
        loss = -mx.mean(mx.take_along_axis(
            log_probs, targets_flat[:, None], axis=1
        ))

        return loss


class GPTModelOp(Op):
    """Forward pass through GPT model."""

    def __init__(self, model, inputs: str, outputs: str) -> None:
        super().__init__([inputs], outputs)
        self.model = model

    def forward(self, data, state):
        tokens = data[0]
        logits, _ = self.model(tokens)
        return logits


class Perplexity(Trace):
    """Track perplexity (exp of average cross-entropy loss)."""

    def __init__(self, loss_key: str = "lm_loss") -> None:
        self.loss_key = loss_key
        self.total_loss = 0.0
        self.count = 0

    def on_epoch_begin(self, state):
        self.total_loss = 0.0
        self.count = 0

    def on_batch_end(self, batch, state):
        if self.loss_key in batch:
            loss = batch[self.loss_key]
            if isinstance(loss, mx.array):
                loss = float(loss.item())
            self.total_loss += loss
            self.count += 1

    def on_epoch_end(self, state):
        avg_loss = self.total_loss / max(1, self.count)
        import math
        state['metrics']['perplexity'] = math.exp(min(avg_loss, 20))  # Clip to avoid overflow
        state['metrics']['lm_loss'] = avg_loss


def create_sequences(tokens: list, seq_length: int, stride: int = None):
    """Create input-target sequence pairs for language modeling."""
    if stride is None:
        stride = seq_length

    inputs = []
    targets = []

    for i in range(0, len(tokens) - seq_length - 1, stride):
        inputs.append(tokens[i:i + seq_length])
        targets.append(tokens[i + 1:i + seq_length + 1])

    return inputs, targets


def get_estimator(
    epochs: int = 10,
    batch_size: int = 64,
    seq_length: int = 128,
    dims: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create GPT language model estimator.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        seq_length: Sequence length for training.
        dims: Model dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        save_dir: Directory to save models.

    Returns:
        Configured Estimator ready for training.
    """
    # Download and prepare data
    text = download_shakespeare()
    tokenizer = CharacterTokenizer(text)
    tokens = tokenizer.encode(text)

    print(f"Dataset: {len(text):,} characters, {tokenizer.vocab_size} unique")

    # Create sequences
    inputs, targets = create_sequences(tokens, seq_length, stride=seq_length // 2)

    # Split train/eval (90/10)
    split = int(len(inputs) * 0.9)
    train_inputs, train_targets = inputs[:split], targets[:split]
    eval_inputs, eval_targets = inputs[split:], targets[split:]

    train_data = MLXDataset({
        "x": mx.array(train_inputs, dtype=mx.int32),
        "y": mx.array(train_targets, dtype=mx.int32)
    })
    eval_data = MLXDataset({
        "x": mx.array(eval_inputs, dtype=mx.int32),
        "y": mx.array(eval_targets, dtype=mx.int32)
    })

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[],  # No preprocessing needed
    )

    # Build GPT model
    model = fe.build(
        model_fn=lambda: GPT(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=seq_length,
            dims=dims,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1
        ),
        optimizer_fn="adam"
    )

    network = fe.Network([
        GPTModelOp(model=model, inputs="x", outputs="logits"),
        LanguageModelLoss(
            inputs=("logits", "y"),
            outputs="lm_loss",
            vocab_size=tokenizer.vocab_size
        ),
        fe.op.UpdateOp(model=model, loss_name="lm_loss")
    ])

    steps_per_epoch = len(train_inputs) // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = steps_per_epoch * 2

    traces = [
        Perplexity(loss_key="lm_loss"),
        ModelSaver(model=model, save_dir=save_dir, frequency=5),
        LRScheduler(
            model=model,
            lr_fn=lambda step: warmup_cosine_decay(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                init_lr=3e-4,
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


def generate_text(
    model,
    tokenizer: CharacterTokenizer,
    prompt: str = "ROMEO:",
    max_tokens: int = 500,
    temperature: float = 0.8
) -> str:
    """Generate text from trained model."""
    tokens = mx.array([tokenizer.encode(prompt)])

    for _ in range(max_tokens):
        logits, _ = model(tokens[:, -model.max_seq_len:])
        next_logits = logits[:, -1, :] / temperature

        # Sample from distribution
        probs = mx.softmax(next_logits, axis=-1)
        next_token = mx.random.categorical(probs, axis=-1)[:, None]
        tokens = mx.concatenate([tokens, next_token], axis=1)

    return tokenizer.decode(tokens[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Language Model with FastMLX")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--dims", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of heads")
    args = parser.parse_args()

    print("GPT Character-Level Language Model")
    print(f"  Dims: {args.dims}, Layers: {args.num_layers}, Heads: {args.num_heads}")

    est = get_estimator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        dims=args.dims,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    est.fit()
    est.test()
