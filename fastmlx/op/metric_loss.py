"""Metric learning loss operations (Triplet, Contrastive, etc.)."""

from __future__ import annotations

from typing import Any, MutableMapping, Tuple

import mlx.core as mx

from .op import LossOp


class TripletLoss(LossOp):
    """Triplet loss for metric learning.

    Minimizes distance between anchor and positive, while maximizing
    distance between anchor and negative samples.

    Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)

    Args:
        inputs: Tuple of (anchor, positive, negative) keys for embeddings.
        outputs: Output key for the loss value.
        margin: Margin for triplet loss.
        p: Norm degree for distance calculation. Default is 2 (Euclidean).
        reduction: How to reduce the loss. Options: 'mean', 'sum', 'none'.

    Example:
        >>> loss_op = TripletLoss(
        ...     inputs=("anchor", "positive", "negative"),
        ...     outputs="loss",
        ...     margin=1.0
        ... )
    """

    def __init__(
        self,
        inputs: Tuple[str, str, str],
        outputs: str,
        margin: float = 1.0,
        p: int = 2,
        reduction: str = "mean"
    ) -> None:
        super().__init__(inputs, outputs)
        self.margin = margin
        self.p = p
        self.reduction = reduction

    def forward(
        self,
        data: Tuple[mx.array, mx.array, mx.array],
        state: MutableMapping[str, Any]
    ) -> mx.array:
        anchor, positive, negative = data

        # Calculate distances
        pos_dist = self._pairwise_distance(anchor, positive)
        neg_dist = self._pairwise_distance(anchor, negative)

        # Calculate triplet loss
        loss = mx.maximum(pos_dist - neg_dist + self.margin, 0.0)

        # Reduction
        if self.reduction == "mean":
            return mx.mean(loss)
        elif self.reduction == "sum":
            return mx.sum(loss)
        else:
            return loss

    def _pairwise_distance(self, x1: mx.array, x2: mx.array) -> mx.array:
        """Calculate pairwise distance between two sets of embeddings."""
        if self.p == 2:
            return mx.sqrt(mx.sum((x1 - x2) ** 2, axis=-1) + 1e-8)
        else:
            return mx.sum(mx.abs(x1 - x2) ** self.p, axis=-1) ** (1.0 / self.p)


class ContrastiveLoss(LossOp):
    """Contrastive loss for siamese networks.

    Pulls similar pairs together and pushes dissimilar pairs apart.

    For similar pairs (label=1): loss = d^2
    For dissimilar pairs (label=0): loss = max(margin - d, 0)^2

    Args:
        inputs: Tuple of (embedding1, embedding2, label) keys.
                Labels should be 1 for similar pairs, 0 for dissimilar.
        outputs: Output key for the loss value.
        margin: Margin for dissimilar pairs.
        reduction: How to reduce the loss. Options: 'mean', 'sum', 'none'.

    Example:
        >>> loss_op = ContrastiveLoss(
        ...     inputs=("emb1", "emb2", "is_similar"),
        ...     outputs="loss",
        ...     margin=2.0
        ... )
    """

    def __init__(
        self,
        inputs: Tuple[str, str, str],
        outputs: str,
        margin: float = 2.0,
        reduction: str = "mean"
    ) -> None:
        super().__init__(inputs, outputs)
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        data: Tuple[mx.array, mx.array, mx.array],
        state: MutableMapping[str, Any]
    ) -> mx.array:
        emb1, emb2, labels = data

        # Calculate Euclidean distance
        dist = mx.sqrt(mx.sum((emb1 - emb2) ** 2, axis=-1) + 1e-8)

        # Ensure labels are float for computation
        labels = labels.astype(mx.float32)

        # Contrastive loss
        # Similar pairs: minimize distance
        # Dissimilar pairs: maximize distance (up to margin)
        pos_loss = labels * dist ** 2
        neg_loss = (1 - labels) * mx.maximum(self.margin - dist, 0.0) ** 2
        loss = pos_loss + neg_loss

        # Reduction
        if self.reduction == "mean":
            return mx.mean(loss)
        elif self.reduction == "sum":
            return mx.sum(loss)
        else:
            return loss


class CenterLoss(LossOp):
    """Center loss for feature learning.

    Minimizes the distance between features and their corresponding
    class centers. Often used with softmax loss for face recognition.

    Args:
        inputs: Tuple of (features, labels) keys.
        outputs: Output key for the loss value.
        num_classes: Number of classes.
        feature_dim: Dimension of the feature vectors.
        alpha: Learning rate for center updates.
        reduction: How to reduce the loss. Options: 'mean', 'sum', 'none'.

    Example:
        >>> loss_op = CenterLoss(
        ...     inputs=("features", "y"),
        ...     outputs="center_loss",
        ...     num_classes=10,
        ...     feature_dim=512
        ... )
    """

    def __init__(
        self,
        inputs: Tuple[str, str],
        outputs: str,
        num_classes: int,
        feature_dim: int,
        alpha: float = 0.5,
        reduction: str = "mean"
    ) -> None:
        super().__init__(inputs, outputs)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.reduction = reduction

        # Initialize centers
        self.centers = mx.zeros((num_classes, feature_dim))

    def forward(
        self,
        data: Tuple[mx.array, mx.array],
        state: MutableMapping[str, Any]
    ) -> mx.array:
        features, labels = data

        # Get centers for each sample
        labels_int = labels.astype(mx.int32)
        if labels_int.ndim > 1:
            labels_int = labels_int.flatten()

        batch_centers = self.centers[labels_int]

        # Calculate loss (distance to centers)
        diff = features - batch_centers
        loss = mx.sum(diff ** 2, axis=-1)

        # Update centers (only during training)
        if state.get("mode") == "train":
            self._update_centers(features, labels_int)

        # Reduction
        if self.reduction == "mean":
            return mx.mean(loss)
        elif self.reduction == "sum":
            return mx.sum(loss)
        else:
            return loss

    def _update_centers(self, features: mx.array, labels: mx.array) -> None:
        """Update class centers based on batch features."""
        # This is a simplified update - in practice you'd want to use
        # gradient descent on the centers as well
        centers_list = []
        for i in range(self.num_classes):
            mask = labels == i
            if mx.sum(mask) > 0:
                class_features = features[mask]
                center_diff = mx.mean(class_features, axis=0) - self.centers[i]
                updated_center = self.centers[i] + self.alpha * center_diff
                centers_list.append(updated_center)
            else:
                centers_list.append(self.centers[i])
        self.centers = mx.stack(centers_list, axis=0)


class CosineSimilarityLoss(LossOp):
    """Cosine similarity loss for embedding learning.

    Optimizes embeddings to have high cosine similarity for similar
    pairs and low similarity for dissimilar pairs.

    Args:
        inputs: Tuple of (embedding1, embedding2, label) keys.
                Labels should be 1 for similar pairs, -1 for dissimilar.
        outputs: Output key for the loss value.
        margin: Margin for the loss.
        reduction: How to reduce the loss. Options: 'mean', 'sum', 'none'.

    Example:
        >>> loss_op = CosineSimilarityLoss(
        ...     inputs=("emb1", "emb2", "label"),
        ...     outputs="loss"
        ... )
    """

    def __init__(
        self,
        inputs: Tuple[str, str, str],
        outputs: str,
        margin: float = 0.0,
        reduction: str = "mean"
    ) -> None:
        super().__init__(inputs, outputs)
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        data: Tuple[mx.array, mx.array, mx.array],
        state: MutableMapping[str, Any]
    ) -> mx.array:
        emb1, emb2, labels = data

        # Normalize embeddings
        emb1_norm = emb1 / (mx.sqrt(mx.sum(emb1 ** 2, axis=-1, keepdims=True)) + 1e-8)
        emb2_norm = emb2 / (mx.sqrt(mx.sum(emb2 ** 2, axis=-1, keepdims=True)) + 1e-8)

        # Calculate cosine similarity
        cos_sim = mx.sum(emb1_norm * emb2_norm, axis=-1)

        # Loss: 1 - y * cos_sim
        # For similar pairs (y=1): minimize 1 - cos_sim (maximize similarity)
        # For dissimilar pairs (y=-1): minimize 1 + cos_sim (minimize similarity)
        labels = labels.astype(mx.float32)
        loss = mx.maximum(0.0, 1.0 - labels * cos_sim + self.margin)

        # Reduction
        if self.reduction == "mean":
            return mx.mean(loss)
        elif self.reduction == "sum":
            return mx.sum(loss)
        else:
            return loss
