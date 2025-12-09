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


class AUC(Trace):
    """Compute Area Under the ROC Curve (AUC) for binary or multi-class classification.

    For binary classification, computes AUC directly.
    For multi-class, uses one-vs-rest (OvR) strategy with specified averaging.

    Args:
        true_key: Key for ground truth labels in batch.
        pred_key: Key for predictions (probabilities) in batch.
        average: Averaging method for multi-class: 'macro', 'weighted', or None (per-class).
        output_name: Name for the metric in state['metrics'].
        mode: Mode(s) in which to run. Defaults to None (all modes).

    Example:
        >>> AUC(true_key="y", pred_key="y_pred", output_name="auc")
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        average: Optional[str] = "macro",
        output_name: str = "auc",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode)
        self.true_key = true_key
        self.pred_key = pred_key
        self.average = average
        self.output_name = output_name
        self.y_true_list: List = []
        self.y_pred_list: List = []

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.y_true_list = []
        self.y_pred_list = []

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]

        # Convert to numpy for accumulation
        if isinstance(y, mx.array):
            y = y.tolist()
        if isinstance(y_pred, mx.array):
            y_pred = y_pred.tolist()

        self.y_true_list.extend(y)
        self.y_pred_list.extend(y_pred)

    def _compute_binary_auc(self, y_true: List, y_scores: List) -> float:
        """Compute AUC for binary classification using trapezoidal rule."""
        # Sort by predicted scores descending
        paired = sorted(zip(y_scores, y_true), reverse=True)

        # Count positives and negatives
        n_pos = sum(1 for _, y in paired if y == 1)
        n_neg = len(paired) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5  # Undefined, return random baseline

        # Compute AUC via Mann-Whitney U statistic
        tp = 0
        fp = 0
        auc = 0.0
        prev_score = float('inf')

        for score, label in paired:
            if score != prev_score:
                prev_score = score
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += tp

        return auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        import numpy as np

        y_true = np.array(self.y_true_list)
        y_pred = np.array(self.y_pred_list)

        # Handle one-hot encoded labels
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true_labels = np.argmax(y_true, axis=-1)
            n_classes = y_true.shape[-1]
        else:
            y_true_labels = y_true.astype(int)
            n_classes = int(np.max(y_true_labels)) + 1 if len(y_true_labels) > 0 else 2

        # Binary classification
        if n_classes == 2:
            if y_pred.ndim > 1 and y_pred.shape[-1] == 2:
                scores = y_pred[:, 1].tolist()
            elif y_pred.ndim == 1:
                scores = y_pred.tolist()
            else:
                scores = y_pred[:, 1].tolist() if y_pred.shape[-1] > 1 else y_pred.flatten().tolist()

            auc = self._compute_binary_auc(y_true_labels.tolist(), scores)
            state['metrics'][self.output_name] = auc
            return

        # Multi-class: one-vs-rest
        if y_pred.ndim == 1:
            # Can't compute multi-class AUC without class probabilities
            state['metrics'][self.output_name] = 0.0
            return

        aucs = []
        weights = []
        for c in range(n_classes):
            binary_true = (y_true_labels == c).astype(int).tolist()
            scores = y_pred[:, c].tolist()

            class_weight = sum(binary_true)
            if class_weight > 0:  # Only compute if class exists
                auc_c = self._compute_binary_auc(binary_true, scores)
                aucs.append(auc_c)
                weights.append(class_weight)

        if not aucs:
            state['metrics'][self.output_name] = 0.0
            return

        if self.average == "macro":
            final_auc = sum(aucs) / len(aucs)
        elif self.average == "weighted":
            total_weight = sum(weights)
            final_auc = sum(a * w for a, w in zip(aucs, weights)) / total_weight if total_weight > 0 else 0.0
        else:
            # Return per-class AUCs
            final_auc = aucs

        state['metrics'][self.output_name] = final_auc


class BLEUScore(Trace):
    """Compute BLEU score for machine translation and text generation.

    BLEU (Bilingual Evaluation Understudy) measures the similarity between
    predicted text and reference text using n-gram precision.

    Args:
        true_key: Key for reference text(s) in batch. Can be string or list of strings.
        pred_key: Key for predicted text in batch.
        n_gram: Maximum n-gram order to consider (default 4 for BLEU-4).
        output_name: Name for the metric in state['metrics'].
        mode: Mode(s) in which to run. Defaults to ["eval", "test"].

    Reference:
        Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine
        Translation", ACL 2002.

    Example:
        >>> BLEUScore(true_key="reference", pred_key="translation", n_gram=4)
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        n_gram: int = 4,
        output_name: str = "bleu",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode if mode else ["eval", "test"])
        self.true_key = true_key
        self.pred_key = pred_key
        self.n_gram = n_gram
        self.output_name = output_name
        # Corpus-level statistics
        self.clip_counts: List[int] = []  # Clipped n-gram counts per order
        self.total_counts: List[int] = []  # Total n-gram counts per order
        self.ref_length: int = 0
        self.hyp_length: int = 0

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.clip_counts = [0] * self.n_gram
        self.total_counts = [0] * self.n_gram
        self.ref_length = 0
        self.hyp_length = 0

    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[tuple, int]:
        """Extract n-grams from token list."""
        ngrams: Dict[tuple, int] = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    def _tokenize(self, text: Union[str, List]) -> List[str]:
        """Simple whitespace tokenization."""
        if isinstance(text, list):
            return [str(t) for t in text]
        return str(text).lower().split()

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        references = batch[self.true_key]
        hypotheses = batch[self.pred_key]

        # Handle batch processing
        if not isinstance(references, (list, tuple)):
            references = [references]
        if not isinstance(hypotheses, (list, tuple)):
            hypotheses = [hypotheses]

        for ref, hyp in zip(references, hypotheses):
            ref_tokens = self._tokenize(ref)
            hyp_tokens = self._tokenize(hyp)

            # Update lengths for brevity penalty
            self.ref_length += len(ref_tokens)
            self.hyp_length += len(hyp_tokens)

            # Compute n-gram statistics
            for n in range(1, self.n_gram + 1):
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                hyp_ngrams = self._get_ngrams(hyp_tokens, n)

                # Clipped count: min of hyp count and ref count for each n-gram
                clipped = 0
                total = 0
                for ngram, count in hyp_ngrams.items():
                    clipped += min(count, ref_ngrams.get(ngram, 0))
                    total += count

                self.clip_counts[n - 1] += clipped
                self.total_counts[n - 1] += total

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        # Compute brevity penalty
        if self.hyp_length == 0:
            state['metrics'][self.output_name] = 0.0
            return

        if self.hyp_length >= self.ref_length:
            bp = 1.0
        else:
            bp = math.exp(1 - self.ref_length / self.hyp_length)

        # Compute geometric mean of modified precisions with smoothing
        log_precisions = []
        for n in range(self.n_gram):
            if self.total_counts[n] == 0:
                # Smoothing: add 1 to avoid log(0)
                precision = 1.0 / (self.hyp_length + 1)
            else:
                # Add-one smoothing for zero counts
                precision = (self.clip_counts[n] + 1) / (self.total_counts[n] + 1)
            log_precisions.append(math.log(precision))

        # Geometric mean with equal weights
        avg_log_precision = sum(log_precisions) / len(log_precisions)
        bleu = bp * math.exp(avg_log_precision)

        state['metrics'][self.output_name] = bleu


class CalibrationError(Trace):
    """Compute Expected Calibration Error (ECE) for classification models.

    ECE measures how well predicted probabilities match actual correctness.
    A well-calibrated model should have predictions with 70% confidence
    being correct 70% of the time.

    Args:
        true_key: Key for ground truth labels in batch.
        pred_key: Key for predicted probabilities in batch.
        n_bins: Number of bins for calibration (default 10).
        method: 'ece' for Expected Calibration Error, 'mce' for Maximum CE.
        output_name: Name for the metric in state['metrics'].
        mode: Mode(s) in which to run. Defaults to ["eval", "test"].

    Reference:
        Naeini et al., "Obtaining Well Calibrated Probabilities Using Bayesian
        Binning into Quantiles", AAAI 2015.

    Example:
        >>> CalibrationError(true_key="y", pred_key="y_pred", n_bins=15)
    """

    def __init__(
        self,
        true_key: str = "y",
        pred_key: str = "y_pred",
        n_bins: int = 10,
        method: str = "ece",
        output_name: str = "calibration_error",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode if mode else ["eval", "test"])
        self.true_key = true_key
        self.pred_key = pred_key
        self.n_bins = n_bins
        self.method = method
        self.output_name = output_name
        self.confidences: List[float] = []
        self.accuracies: List[int] = []

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.confidences = []
        self.accuracies = []

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        import numpy as np

        y = batch[self.true_key]
        y_pred = batch[self.pred_key]

        if isinstance(y, mx.array):
            y = np.array(y)
        if isinstance(y_pred, mx.array):
            y_pred = np.array(y_pred)

        # Handle one-hot labels
        if y.ndim > 1 and y.shape[-1] > 1:
            y = np.argmax(y, axis=-1)

        # Get predicted class and confidence
        if y_pred.ndim > 1:
            pred_class = np.argmax(y_pred, axis=-1)
            confidence = np.max(y_pred, axis=-1)
        else:
            pred_class = (y_pred > 0.5).astype(int)
            confidence = np.where(pred_class == 1, y_pred, 1 - y_pred)

        # Record confidence and correctness
        correct = (pred_class == y).astype(int)
        self.confidences.extend(confidence.tolist())
        self.accuracies.extend(correct.tolist())

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        import numpy as np

        if not self.confidences:
            state['metrics'][self.output_name] = 0.0
            return

        confidences = np.array(self.confidences)
        accuracies = np.array(self.accuracies)

        # Bin by confidence
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_errors = []

        for i in range(self.n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                bin_error = abs(bin_accuracy - bin_confidence)
                bin_errors.append((bin_error, bin_size))

        if not bin_errors:
            state['metrics'][self.output_name] = 0.0
            return

        if self.method == "mce":
            # Maximum Calibration Error
            error = max(e for e, _ in bin_errors)
        else:
            # Expected Calibration Error (weighted by bin size)
            total = sum(s for _, s in bin_errors)
            error = sum(e * s for e, s in bin_errors) / total

        state['metrics'][self.output_name] = error


class MeanAveragePrecision(Trace):
    """Compute Mean Average Precision (mAP) for object detection.

    Calculates mAP across multiple IoU thresholds following COCO evaluation.

    Args:
        true_key: Key for ground truth boxes [x1, y1, w, h, class_id].
        pred_key: Key for predictions [x1, y1, w, h, class_id, score].
        iou_thresholds: List of IoU thresholds (default: 0.5 to 0.95).
        output_name: Name for the metric in state['metrics'].
        mode: Mode(s) in which to run. Defaults to ["eval", "test"].

    Example:
        >>> MeanAveragePrecision(true_key="boxes", pred_key="detections")
    """

    def __init__(
        self,
        true_key: str = "boxes",
        pred_key: str = "detections",
        iou_thresholds: Optional[List[float]] = None,
        output_name: str = "mAP",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(inputs=[true_key, pred_key], outputs=[output_name], mode=mode if mode else ["eval", "test"])
        self.true_key = true_key
        self.pred_key = pred_key
        self.iou_thresholds = iou_thresholds or [0.5 + 0.05 * i for i in range(10)]
        self.output_name = output_name
        self.all_detections: List = []
        self.all_ground_truths: List = []
        self.image_id: int = 0

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.all_detections = []
        self.all_ground_truths = []
        self.image_id = 0

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes [x1, y1, w, h]."""
        x1_1, y1_1, w1, h1 = box1[:4]
        x1_2, y1_2, w2, h2 = box2[:4]

        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        intersection = inter_w * inter_h

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / (union + 1e-8)

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        import numpy as np

        gt_boxes = batch[self.true_key]
        pred_boxes = batch[self.pred_key]

        if isinstance(gt_boxes, mx.array):
            gt_boxes = np.array(gt_boxes)
        if isinstance(pred_boxes, mx.array):
            pred_boxes = np.array(pred_boxes)

        # Handle batch dimension
        if gt_boxes.ndim == 2:
            gt_boxes = [gt_boxes]
            pred_boxes = [pred_boxes]

        for gt, pred in zip(gt_boxes, pred_boxes):
            # Filter out padding (boxes with all zeros)
            gt = [b for b in gt if np.any(b[:4] != 0)]
            pred = [b for b in pred if np.any(b[:4] != 0)]

            self.all_ground_truths.append({
                'image_id': self.image_id,
                'boxes': gt
            })
            self.all_detections.append({
                'image_id': self.image_id,
                'boxes': pred
            })
            self.image_id += 1

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        import numpy as np

        if not self.all_ground_truths or not self.all_detections:
            state['metrics'][self.output_name] = 0.0
            state['metrics']['AP50'] = 0.0
            state['metrics']['AP75'] = 0.0
            return

        # Collect all classes
        all_classes = set()
        for gt in self.all_ground_truths:
            for box in gt['boxes']:
                all_classes.add(int(box[4]))
        for det in self.all_detections:
            for box in det['boxes']:
                all_classes.add(int(box[4]))

        aps_per_threshold = {t: [] for t in self.iou_thresholds}

        for class_id in all_classes:
            # Gather all detections and GTs for this class
            class_dets = []
            class_gts = {}
            n_gt = 0

            for img_idx, (gt_data, det_data) in enumerate(zip(self.all_ground_truths, self.all_detections)):
                # Ground truths for this image/class
                gt_boxes = [b for b in gt_data['boxes'] if int(b[4]) == class_id]
                class_gts[img_idx] = {'boxes': gt_boxes, 'matched': [False] * len(gt_boxes)}
                n_gt += len(gt_boxes)

                # Detections for this image/class
                for box in det_data['boxes']:
                    if int(box[4]) == class_id:
                        class_dets.append({
                            'image_id': img_idx,
                            'box': box[:4],
                            'score': box[5] if len(box) > 5 else 1.0
                        })

            if n_gt == 0:
                continue

            # Sort detections by score
            class_dets.sort(key=lambda x: x['score'], reverse=True)

            for iou_thresh in self.iou_thresholds:
                # Reset matched flags
                for img_idx in class_gts:
                    class_gts[img_idx]['matched'] = [False] * len(class_gts[img_idx]['boxes'])

                tp = np.zeros(len(class_dets))
                fp = np.zeros(len(class_dets))

                for det_idx, det in enumerate(class_dets):
                    img_idx = det['image_id']
                    gt_boxes = class_gts[img_idx]['boxes']
                    matched = class_gts[img_idx]['matched']

                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if matched[gt_idx]:
                            continue
                        iou = self._compute_iou(det['box'], gt_box[:4])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_thresh:
                        tp[det_idx] = 1
                        class_gts[img_idx]['matched'][best_gt_idx] = True
                    else:
                        fp[det_idx] = 1

                # Compute precision-recall curve
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                recall = tp_cumsum / n_gt
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

                # Compute AP using 11-point interpolation
                ap = 0
                for r in np.linspace(0, 1, 11):
                    prec_at_recall = precision[recall >= r]
                    if len(prec_at_recall) > 0:
                        ap += np.max(prec_at_recall) / 11

                aps_per_threshold[iou_thresh].append(ap)

        # Compute mAP
        mAP = np.mean([np.mean(aps) if aps else 0 for aps in aps_per_threshold.values()])
        AP50 = np.mean(aps_per_threshold.get(0.5, [0]))
        AP75 = np.mean(aps_per_threshold.get(0.75, [0]))

        state['metrics'][self.output_name] = float(mAP)
        state['metrics']['AP50'] = float(AP50)
        state['metrics']['AP75'] = float(AP75)
