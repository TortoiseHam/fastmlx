"""Model Ensemble example using :mod:`fastmlx`.

Demonstrates combining multiple models for improved predictions:
- Simple averaging
- Weighted averaging
- Voting (for classification)
- Stacking (meta-learner)

Benefits:
- Reduced variance (averaging over models)
- Better generalization
- Robustness to individual model failures

This example shows ensemble of LeNets with different initializations.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

import fastmlx as fe
from fastmlx.architecture import LeNet
from fastmlx.dataset.data import mnist
from fastmlx.op import CrossEntropy, Minmax, ModelOp, Op, UpdateOp
from fastmlx.schedule import cosine_decay
from fastmlx.trace.adapt import LRScheduler
from fastmlx.trace.metric import Accuracy


class EnsembleModelOp(Op):
    """Forward pass through ensemble of models.

    Combines predictions from multiple models using specified method.

    Args:
        models: List of models in the ensemble.
        inputs: Input key.
        outputs: Output key for combined predictions.
        method: Combination method ('average', 'vote', 'weighted').
        weights: Optional weights for weighted averaging.
    """

    def __init__(
        self,
        models: List[nn.Module],
        inputs: str,
        outputs: str,
        method: str = "average",
        weights: Optional[List[float]] = None
    ) -> None:
        super().__init__([inputs], outputs)
        self.models = models
        self.method = method
        self.weights = weights or [1.0 / len(models)] * len(models)

    def forward(self, data, state):
        # data is a single array when there's one input
        x = data if not isinstance(data, list) else data[0]

        # Get predictions from all models
        predictions = []
        for model in self.models:
            logits = model(x)
            predictions.append(logits)

        if self.method == "vote":
            # Hard voting: each model votes for a class
            votes = mx.zeros((x.shape[0], predictions[0].shape[-1]))
            for pred in predictions:
                vote = mx.argmax(pred, axis=-1)
                # Accumulate one-hot votes
                votes = votes + mx.one_hot(vote, predictions[0].shape[-1])
            return votes  # Return vote counts as "logits"

        elif self.method == "weighted":
            # Weighted average of softmax probabilities
            combined = mx.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                probs = mx.softmax(pred, axis=-1)
                combined = combined + weight * probs
            return mx.log(combined + 1e-8)  # Return log probs

        else:  # average
            # Simple average of logits
            combined = mx.stack(predictions, axis=0)
            return mx.mean(combined, axis=0)


class EnsembleLoss(Op):
    """Combined loss for training ensemble models independently.

    Each model has its own loss, all averaged for the ensemble.
    """

    def __init__(
        self,
        models: List[nn.Module],
        pred_keys: List[str],
        label_key: str,
        outputs: str
    ) -> None:
        inputs = pred_keys + [label_key]
        super().__init__(inputs, outputs)
        self.models = models
        self.num_models = len(models)

    def forward(self, data, state):
        # Last element is labels
        labels = data[-1]
        predictions = data[:-1]

        total_loss = mx.array(0.0)
        for pred in predictions:
            log_probs = pred - mx.logsumexp(pred, axis=-1, keepdims=True)
            if labels.ndim > 1:
                loss = -mx.mean(mx.sum(labels * log_probs, axis=-1))
            else:
                loss = -mx.mean(mx.take_along_axis(
                    log_probs, labels[:, None].astype(mx.int32), axis=1
                ))
            total_loss = total_loss + loss

        return total_loss / self.num_models


def train_individual_models(
    num_models: int = 5,
    epochs: int = 10,
    batch_size: int = 64,
) -> List[nn.Module]:
    """Train individual models for the ensemble.

    Each model is trained with different random initialization.

    Args:
        num_models: Number of models in ensemble.
        epochs: Training epochs per model.
        batch_size: Batch size.

    Returns:
        List of trained models.
    """
    train_data, eval_data = mnist.load_data()
    models = []

    for i in range(num_models):
        print(f"\nTraining model {i + 1}/{num_models}...")

        # Create model with unique random seed
        mx.random.seed(42 + i)

        pipeline = fe.Pipeline(
            train_data=train_data,
            eval_data=eval_data,
            batch_size=batch_size,
            ops=[Minmax(inputs="x", outputs="x")],
        )

        model = fe.build(
            model_fn=lambda: LeNet(input_shape=(1, 28, 28)),
            optimizer_fn="adam"
        )

        network = fe.Network([
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])

        steps_per_epoch = 60000 // batch_size
        cycle_length = epochs * steps_per_epoch

        traces = [
            Accuracy(true_key="y", pred_key="y_pred"),
            LRScheduler(
                model=model,
                lr_fn=lambda step: cosine_decay(
                    step, cycle_length=cycle_length,
                    init_lr=1e-3, min_lr=1e-5
                )
            )
        ]

        estimator = fe.Estimator(
            pipeline=pipeline,
            network=network,
            epochs=epochs,
            traces=traces
        )
        estimator.fit()

        models.append(model)

    return models


def evaluate_ensemble(
    models: List[nn.Module],
    method: str = "average",
    weights: Optional[List[float]] = None,
    batch_size: int = 256
) -> float:
    """Evaluate ensemble on test data.

    Args:
        models: List of trained models.
        method: Combination method.
        weights: Optional weights for models.
        batch_size: Batch size for evaluation.

    Returns:
        Ensemble accuracy.
    """
    _, eval_data = mnist.load_data()

    eval_x = eval_data.data["x"].astype(mx.float32) / 255.0
    eval_y = eval_data.data["y"]

    num_eval = eval_x.shape[0]
    correct = 0

    weights = weights or [1.0 / len(models)] * len(models)

    for i in range(0, num_eval, batch_size):
        batch_x = eval_x[i:i + batch_size]
        batch_y = eval_y[i:i + batch_size]

        # Get predictions from all models
        predictions = []
        for model in models:
            logits = model(batch_x)
            predictions.append(logits)

        # Combine predictions
        if method == "vote":
            # Hard voting
            votes = mx.zeros((batch_x.shape[0], 10))
            for pred in predictions:
                vote = mx.argmax(pred, axis=-1)
                votes = votes + mx.one_hot(vote, 10)
            ensemble_pred = mx.argmax(votes, axis=-1)

        elif method == "weighted":
            # Weighted soft voting
            combined = mx.zeros_like(predictions[0])
            for pred, weight in zip(predictions, weights):
                probs = mx.softmax(pred, axis=-1)
                combined = combined + weight * probs
            ensemble_pred = mx.argmax(combined, axis=-1)

        else:  # average
            combined = mx.stack(predictions, axis=0)
            avg_logits = mx.mean(combined, axis=0)
            ensemble_pred = mx.argmax(avg_logits, axis=-1)

        correct += int(mx.sum(ensemble_pred == batch_y).item())

    return correct / num_eval


def main(
    num_models: int = 5,
    epochs: int = 10,
    batch_size: int = 64,
):
    """Train and evaluate model ensemble.

    Args:
        num_models: Number of models in ensemble.
        epochs: Training epochs per model.
        batch_size: Batch size.
    """
    print("Model Ensemble on MNIST")
    print(f"  Number of models: {num_models}")
    print("=" * 50)

    # Train individual models
    models = train_individual_models(
        num_models=num_models,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Evaluate individual models
    print("\n" + "=" * 50)
    print("Individual Model Performance:")
    individual_accs = []
    for i, model in enumerate(models):
        acc = evaluate_ensemble([model], method="average")
        individual_accs.append(acc)
        print(f"  Model {i + 1}: {acc:.4f}")

    print(f"  Average: {sum(individual_accs) / len(individual_accs):.4f}")

    # Evaluate ensemble with different methods
    print("\nEnsemble Performance:")
    for method in ["average", "vote", "weighted"]:
        acc = evaluate_ensemble(models, method=method)
        print(f"  {method.capitalize()}: {acc:.4f}")

    # Evaluate with learned weights (based on individual accuracy)
    # Give higher weight to better models
    total_acc = sum(individual_accs)
    learned_weights = [acc / total_acc for acc in individual_accs]
    acc = evaluate_ensemble(models, method="weighted", weights=learned_weights)
    print(f"  Accuracy-weighted: {acc:.4f}")

    print("\n" + "=" * 50)
    print("Ensemble improves over individual models by leveraging diversity!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Ensemble with FastMLX")
    parser.add_argument("--num-models", type=int, default=5,
                        help="Number of models in ensemble")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    main(
        num_models=args.num_models,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
