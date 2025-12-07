"""Metric traces used during training."""

from __future__ import annotations

import math
from typing import List, MutableMapping, Optional, Union

import mlx.core as mx

from .base import Trace


class Accuracy(Trace):
    """Compute classification accuracy over an epoch.

    Args:
        true_key: Key for ground truth labels in batch.
        pred_key: Key for predictions in batch.
        output_name: Name for the metric in state['metrics']. Defaults to 'accuracy'.
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        output_name: str = "accuracy",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.output_name = output_name
        self.correct: int = 0
        self.total: int = 0

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.correct = 0
        self.total = 0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]
        # ``y`` may be provided either as integer labels or one-hot vectors.
        # Convert to integer labels to simplify the equality check.
        pred = mx.argmax(y_pred, axis=-1)
        true = mx.argmax(y, axis=-1) if y.ndim > 1 else y
        self.correct += int(mx.sum(pred == true).item())
        self.total += true.shape[0]

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        state['metrics'][self.output_name] = self.correct / max(1, self.total)


class LossMonitor(Trace):
    """Track the mean of a given loss value over an epoch.

    Args:
        loss_key: Key for the loss value in batch.
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        loss_key: str = "ce",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[loss_key], outputs=[loss_key], mode=mode)
        self.loss_key = loss_key
        self.total: float = 0.0
        self.count: int = 0

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.total = 0.0
        self.count = 0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        loss = batch.get(self.loss_key)
        if loss is None:
            return
        if isinstance(loss, mx.array):
            loss = float(loss.item())
        self.total += float(loss)
        self.count += 1

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        if self.count:
            state['metrics'][self.loss_key] = self.total / self.count


class Precision(Trace):
    """Compute precision (positive predictive value) for classification.

    Precision = TP / (TP + FP)

    Args:
        true_key: Key for ground truth labels in batch.
        pred_key: Key for predictions in batch.
        num_classes: Number of classes. If None, inferred from predictions.
        average: Averaging method. 'micro', 'macro', or 'weighted'.
        output_name: Name for the metric in state['metrics'].
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        num_classes: Optional[int] = None,
        average: str = "macro",
        output_name: str = "precision",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.num_classes = num_classes
        self.average = average
        self.output_name = output_name
        self.true_positives: List[int] = []
        self.false_positives: List[int] = []
        self.support: List[int] = []

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        if self.num_classes:
            self.true_positives = [0] * self.num_classes
            self.false_positives = [0] * self.num_classes
            self.support = [0] * self.num_classes
        else:
            self.true_positives = []
            self.false_positives = []
            self.support = []

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]

        pred = mx.argmax(y_pred, axis=-1)
        true = mx.argmax(y, axis=-1) if y.ndim > 1 else y

        # Infer num_classes if needed
        if not self.true_positives:
            n_classes = self.num_classes or int(y_pred.shape[-1])
            self.true_positives = [0] * n_classes
            self.false_positives = [0] * n_classes
            self.support = [0] * n_classes

        pred_np = pred.tolist()
        true_np = true.tolist()

        for p, t in zip(pred_np, true_np):
            p, t = int(p), int(t)
            if p == t:
                self.true_positives[p] += 1
            else:
                self.false_positives[p] += 1
            self.support[t] += 1

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        if self.average == "micro":
            tp_sum = sum(self.true_positives)
            fp_sum = sum(self.false_positives)
            precision = tp_sum / max(1, tp_sum + fp_sum)
        elif self.average == "macro":
            precisions = []
            for tp, fp in zip(self.true_positives, self.false_positives):
                if tp + fp > 0:
                    precisions.append(tp / (tp + fp))
            precision = sum(precisions) / max(1, len(precisions))
        else:  # weighted
            weighted_sum = 0.0
            total_support = sum(self.support)
            for tp, fp, sup in zip(self.true_positives, self.false_positives, self.support):
                if tp + fp > 0:
                    weighted_sum += (tp / (tp + fp)) * sup
            precision = weighted_sum / max(1, total_support)

        state['metrics'][self.output_name] = precision


class Recall(Trace):
    """Compute recall (sensitivity, true positive rate) for classification.

    Recall = TP / (TP + FN)

    Args:
        true_key: Key for ground truth labels in batch.
        pred_key: Key for predictions in batch.
        num_classes: Number of classes. If None, inferred from predictions.
        average: Averaging method. 'micro', 'macro', or 'weighted'.
        output_name: Name for the metric in state['metrics'].
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        num_classes: Optional[int] = None,
        average: str = "macro",
        output_name: str = "recall",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.num_classes = num_classes
        self.average = average
        self.output_name = output_name
        self.true_positives: List[int] = []
        self.false_negatives: List[int] = []
        self.support: List[int] = []

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        if self.num_classes:
            self.true_positives = [0] * self.num_classes
            self.false_negatives = [0] * self.num_classes
            self.support = [0] * self.num_classes
        else:
            self.true_positives = []
            self.false_negatives = []
            self.support = []

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]

        pred = mx.argmax(y_pred, axis=-1)
        true = mx.argmax(y, axis=-1) if y.ndim > 1 else y

        # Infer num_classes if needed
        if not self.true_positives:
            n_classes = self.num_classes or int(y_pred.shape[-1])
            self.true_positives = [0] * n_classes
            self.false_negatives = [0] * n_classes
            self.support = [0] * n_classes

        pred_np = pred.tolist()
        true_np = true.tolist()

        for p, t in zip(pred_np, true_np):
            p, t = int(p), int(t)
            if p == t:
                self.true_positives[t] += 1
            else:
                self.false_negatives[t] += 1
            self.support[t] += 1

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        if self.average == "micro":
            tp_sum = sum(self.true_positives)
            fn_sum = sum(self.false_negatives)
            recall = tp_sum / max(1, tp_sum + fn_sum)
        elif self.average == "macro":
            recalls = []
            for tp, fn in zip(self.true_positives, self.false_negatives):
                if tp + fn > 0:
                    recalls.append(tp / (tp + fn))
            recall = sum(recalls) / max(1, len(recalls))
        else:  # weighted
            weighted_sum = 0.0
            total_support = sum(self.support)
            for tp, fn, sup in zip(self.true_positives, self.false_negatives, self.support):
                if tp + fn > 0:
                    weighted_sum += (tp / (tp + fn)) * sup
            recall = weighted_sum / max(1, total_support)

        state['metrics'][self.output_name] = recall


class F1Score(Trace):
    """Compute F1 score (harmonic mean of precision and recall).

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        true_key: Key for ground truth labels in batch.
        pred_key: Key for predictions in batch.
        num_classes: Number of classes. If None, inferred from predictions.
        average: Averaging method. 'micro', 'macro', or 'weighted'.
        output_name: Name for the metric in state['metrics'].
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        num_classes: Optional[int] = None,
        average: str = "macro",
        output_name: str = "f1_score",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.num_classes = num_classes
        self.average = average
        self.output_name = output_name
        self.true_positives: List[int] = []
        self.false_positives: List[int] = []
        self.false_negatives: List[int] = []
        self.support: List[int] = []

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        if self.num_classes:
            self.true_positives = [0] * self.num_classes
            self.false_positives = [0] * self.num_classes
            self.false_negatives = [0] * self.num_classes
            self.support = [0] * self.num_classes
        else:
            self.true_positives = []
            self.false_positives = []
            self.false_negatives = []
            self.support = []

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]

        pred = mx.argmax(y_pred, axis=-1)
        true = mx.argmax(y, axis=-1) if y.ndim > 1 else y

        # Infer num_classes if needed
        if not self.true_positives:
            n_classes = self.num_classes or int(y_pred.shape[-1])
            self.true_positives = [0] * n_classes
            self.false_positives = [0] * n_classes
            self.false_negatives = [0] * n_classes
            self.support = [0] * n_classes

        pred_np = pred.tolist()
        true_np = true.tolist()

        for p, t in zip(pred_np, true_np):
            p, t = int(p), int(t)
            if p == t:
                self.true_positives[p] += 1
            else:
                self.false_positives[p] += 1
                self.false_negatives[t] += 1
            self.support[t] += 1

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        if self.average == "micro":
            tp_sum = sum(self.true_positives)
            fp_sum = sum(self.false_positives)
            fn_sum = sum(self.false_negatives)
            precision = tp_sum / max(1, tp_sum + fp_sum)
            recall = tp_sum / max(1, tp_sum + fn_sum)
            f1 = 2 * precision * recall / max(1e-7, precision + recall)
        elif self.average == "macro":
            f1_scores = []
            for tp, fp, fn in zip(self.true_positives, self.false_positives, self.false_negatives):
                if tp + fp + fn > 0:
                    precision = tp / max(1, tp + fp)
                    recall = tp / max(1, tp + fn)
                    if precision + recall > 0:
                        f1_scores.append(2 * precision * recall / (precision + recall))
            f1 = sum(f1_scores) / max(1, len(f1_scores))
        else:  # weighted
            weighted_sum = 0.0
            total_support = sum(self.support)
            for tp, fp, fn, sup in zip(self.true_positives, self.false_positives, self.false_negatives, self.support):
                if tp + fp + fn > 0:
                    precision = tp / max(1, tp + fp)
                    recall = tp / max(1, tp + fn)
                    if precision + recall > 0:
                        weighted_sum += (2 * precision * recall / (precision + recall)) * sup
            f1 = weighted_sum / max(1, total_support)

        state['metrics'][self.output_name] = f1


class ConfusionMatrix(Trace):
    """Compute and store confusion matrix for classification.

    Args:
        true_key: Key for ground truth labels in batch.
        pred_key: Key for predictions in batch.
        num_classes: Number of classes. Required.
        output_name: Name for the metric in state['metrics'].
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        num_classes: int = 10,
        output_name: str = "confusion_matrix",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.num_classes = num_classes
        self.output_name = output_name
        self.matrix: List[List[int]] = []

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.matrix = [[0] * self.num_classes for _ in range(self.num_classes)]

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]

        pred = mx.argmax(y_pred, axis=-1)
        true = mx.argmax(y, axis=-1) if y.ndim > 1 else y

        pred_np = pred.tolist()
        true_np = true.tolist()

        for p, t in zip(pred_np, true_np):
            p, t = int(p), int(t)
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.matrix[t][p] += 1

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        state['metrics'][self.output_name] = self.matrix


class MCC(Trace):
    """Compute Matthews Correlation Coefficient for classification.

    MCC is a balanced measure that can be used even when classes are imbalanced.
    Returns a value between -1 and +1. +1 is perfect prediction, 0 is random,
    -1 is total disagreement.

    Args:
        true_key: Key for ground truth labels in batch.
        pred_key: Key for predictions in batch.
        num_classes: Number of classes. If None, inferred from predictions.
        output_name: Name for the metric in state['metrics'].
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        num_classes: Optional[int] = None,
        output_name: str = "mcc",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.num_classes = num_classes
        self.output_name = output_name
        self.matrix: List[List[int]] = []

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        if self.num_classes:
            self.matrix = [[0] * self.num_classes for _ in range(self.num_classes)]
        else:
            self.matrix = []

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]

        pred = mx.argmax(y_pred, axis=-1)
        true = mx.argmax(y, axis=-1) if y.ndim > 1 else y

        # Infer num_classes if needed
        if not self.matrix:
            n_classes = self.num_classes or int(y_pred.shape[-1])
            self.matrix = [[0] * n_classes for _ in range(n_classes)]

        pred_np = pred.tolist()
        true_np = true.tolist()

        for p, t in zip(pred_np, true_np):
            p, t = int(p), int(t)
            n_classes = len(self.matrix)
            if 0 <= t < n_classes and 0 <= p < n_classes:
                self.matrix[t][p] += 1

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        # Multi-class MCC using the confusion matrix
        n = len(self.matrix)
        if n == 0:
            state['metrics'][self.output_name] = 0.0
            return

        # Calculate row sums, column sums, and total
        t_k = [sum(self.matrix[k]) for k in range(n)]  # true class totals
        p_k = [sum(self.matrix[i][k] for i in range(n)) for k in range(n)]  # predicted class totals
        c = sum(self.matrix[k][k] for k in range(n))  # correct predictions
        s = sum(t_k)  # total samples

        if s == 0:
            state['metrics'][self.output_name] = 0.0
            return

        # MCC formula for multi-class
        numerator = c * s - sum(t_k[k] * p_k[k] for k in range(n))
        denom1 = s * s - sum(p_k[k] * p_k[k] for k in range(n))
        denom2 = s * s - sum(t_k[k] * t_k[k] for k in range(n))

        if denom1 == 0 or denom2 == 0:
            state['metrics'][self.output_name] = 0.0
            return

        mcc = numerator / math.sqrt(denom1 * denom2)
        state['metrics'][self.output_name] = mcc


class Dice(Trace):
    """Compute Dice coefficient (F1 for segmentation) over an epoch.

    Dice = 2 * |X intersect Y| / (|X| + |Y|)

    Args:
        true_key: Key for ground truth segmentation mask in batch.
        pred_key: Key for predicted segmentation mask in batch.
        threshold: Threshold for binary prediction (if predictions are probabilities).
        smooth: Smoothing factor to avoid division by zero.
        output_name: Name for the metric in state['metrics'].
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        threshold: float = 0.5,
        smooth: float = 1.0,
        output_name: str = "dice",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.threshold = threshold
        self.smooth = smooth
        self.output_name = output_name
        self.intersection: float = 0.0
        self.union: float = 0.0

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.intersection = 0.0
        self.union = 0.0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]

        # Apply threshold if predictions are probabilities
        if mx.any(y_pred < 0) or mx.any(y_pred > 1):
            y_pred = mx.sigmoid(y_pred)
        y_pred_binary = (y_pred > self.threshold).astype(mx.float32)

        # Flatten for computation
        y_flat = y.reshape(-1)
        pred_flat = y_pred_binary.reshape(-1)

        self.intersection += float(mx.sum(y_flat * pred_flat).item())
        self.union += float(mx.sum(y_flat).item()) + float(mx.sum(pred_flat).item())

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        dice = (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)
        state['metrics'][self.output_name] = dice
