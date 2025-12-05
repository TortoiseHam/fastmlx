"""TensorBoard logging trace for visualization and experiment tracking."""

from __future__ import annotations

import os
from typing import Any, Dict, List, MutableMapping, Optional, Union

from .base import Trace

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TensorBoardLogger(Trace):
    """Log training metrics and model information to TensorBoard.

    This trace logs scalars (loss, accuracy, learning rate), histograms
    (weight distributions), and images to TensorBoard for visualization.

    Args:
        log_dir: Directory to save TensorBoard event files.
        update_freq: How often to log (in batches). If 'epoch', log only at epoch end.
        write_graph: Whether to write the model graph. Currently not supported.
        write_histograms: Whether to write weight histograms.
        histogram_freq: How often to write histograms (in epochs).
        write_images: Whether to write sample images.
        image_freq: How often to write images (in epochs).
        image_keys: Keys in batch containing images to log.
        scalar_keys: Additional keys in batch to log as scalars.
        comment: Suffix for log directory name.

    Example:
        >>> from fastmlx.trace import TensorBoardLogger
        >>>
        >>> # Basic usage
        >>> logger = TensorBoardLogger(log_dir="runs/experiment_1")
        >>> estimator = Estimator(..., traces=[logger])
        >>>
        >>> # With histograms and images
        >>> logger = TensorBoardLogger(
        ...     log_dir="runs/experiment_2",
        ...     write_histograms=True,
        ...     histogram_freq=5,
        ...     write_images=True,
        ...     image_keys=["x"]
        ... )

    Note:
        Requires TensorBoard to be installed: ``pip install tensorboard``
    """

    def __init__(
        self,
        log_dir: str = "runs",
        update_freq: Union[int, str] = "epoch",
        write_graph: bool = False,
        write_histograms: bool = False,
        histogram_freq: int = 1,
        write_images: bool = False,
        image_freq: int = 1,
        image_keys: Optional[List[str]] = None,
        scalar_keys: Optional[List[str]] = None,
        comment: str = ""
    ) -> None:
        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard is not installed. Install it with: pip install tensorboard"
            )

        self.log_dir = log_dir
        self.update_freq = update_freq
        self.write_graph = write_graph
        self.write_histograms = write_histograms
        self.histogram_freq = histogram_freq
        self.write_images = write_images
        self.image_freq = image_freq
        self.image_keys = image_keys or ["x"]
        self.scalar_keys = scalar_keys or []
        self.comment = comment

        self.writer: Optional[SummaryWriter] = None
        self.global_step: int = 0
        self.batch_count: int = 0
        self.model = None

    def on_start(self, state: MutableMapping[str, object]) -> None:
        """Initialize TensorBoard writer."""
        # Create unique log directory
        log_path = self.log_dir
        if self.comment:
            log_path = os.path.join(log_path, self.comment)
        os.makedirs(log_path, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_path)
        print(f"FastMLX-TensorBoard: Logging to {log_path}")

        # Store model reference if available
        if "model" in state:
            self.model = state["model"]

    def on_batch_end(
        self,
        batch: MutableMapping[str, object],
        state: MutableMapping[str, object]
    ) -> None:
        """Log batch-level metrics."""
        if self.writer is None:
            return

        self.global_step += 1
        self.batch_count += 1
        mode = state.get("mode", "train")

        # Only log at specified frequency
        if isinstance(self.update_freq, int) and self.update_freq > 0:
            if self.batch_count % self.update_freq != 0:
                return

            # Log loss
            self._log_scalar_from_batch(batch, "loss", f"{mode}/loss")
            self._log_scalar_from_batch(batch, "ce", f"{mode}/cross_entropy")

            # Log custom scalar keys
            for key in self.scalar_keys:
                self._log_scalar_from_batch(batch, key, f"{mode}/{key}")

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        """Reset batch counter at epoch start."""
        self.batch_count = 0

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        """Log epoch-level metrics and optional histograms/images."""
        if self.writer is None:
            return

        epoch = state.get("epoch", 0)
        mode = state.get("mode", "train")
        metrics = state.get("metrics", {})
        batch = state.get("batch", {})

        # Log all metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{mode}/{key}", value, epoch)
            elif hasattr(value, "item"):
                self.writer.add_scalar(f"{mode}/{key}", float(value.item()), epoch)

        # Log learning rate if available
        lr = self._get_learning_rate(state)
        if lr is not None:
            self.writer.add_scalar("train/learning_rate", lr, epoch)

        # Log histograms if enabled
        if (
            self.write_histograms
            and mode == "train"
            and (epoch + 1) % self.histogram_freq == 0
        ):
            self._log_histograms(epoch)

        # Log images if enabled
        if (
            self.write_images
            and mode == "train"
            and (epoch + 1) % self.image_freq == 0
        ):
            self._log_images(batch, epoch)

        # Flush to disk
        self.writer.flush()

    def on_finish(self, state: MutableMapping[str, object]) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            print("FastMLX-TensorBoard: Writer closed")

    def _log_scalar_from_batch(
        self,
        batch: MutableMapping[str, object],
        key: str,
        tag: str
    ) -> None:
        """Log a scalar value from the batch."""
        value = batch.get(key)
        if value is None:
            return

        if hasattr(value, "item"):
            value = float(value.item())
        elif not isinstance(value, (int, float)):
            return

        self.writer.add_scalar(tag, value, self.global_step)

    def _get_learning_rate(self, state: MutableMapping[str, object]) -> Optional[float]:
        """Extract learning rate from state or model."""
        # Try to get from state
        if "lr" in state:
            return float(state["lr"])

        # Try to get from model's optimizer
        if self.model is not None and hasattr(self.model, "optimizer"):
            lr = getattr(self.model.optimizer, "learning_rate", None)
            if lr is not None:
                return float(lr) if isinstance(lr, (int, float)) else None

        return None

    def _log_histograms(self, epoch: int) -> None:
        """Log weight histograms for all model parameters."""
        if self.model is None:
            return

        try:
            import mlx.core as mx

            # Get model parameters
            if hasattr(self.model, "parameters"):
                params = self.model.parameters()
                self._log_param_dict(params, "", epoch)
        except Exception as e:
            print(f"FastMLX-TensorBoard: Error logging histograms: {e}")

    def _log_param_dict(
        self,
        params: Dict[str, Any],
        prefix: str,
        epoch: int
    ) -> None:
        """Recursively log parameter dictionaries."""
        import mlx.core as mx

        for name, value in params.items():
            full_name = f"{prefix}/{name}" if prefix else name

            if isinstance(value, dict):
                self._log_param_dict(value, full_name, epoch)
            elif isinstance(value, mx.array):
                # Convert MLX array to numpy for TensorBoard
                try:
                    import numpy as np
                    np_array = np.array(value.tolist())
                    self.writer.add_histogram(full_name, np_array, epoch)
                except Exception:
                    pass

    def _log_images(
        self,
        batch: MutableMapping[str, object],
        epoch: int
    ) -> None:
        """Log sample images from the batch."""
        import mlx.core as mx

        for key in self.image_keys:
            images = batch.get(key)
            if images is None:
                continue

            try:
                import numpy as np

                # Convert MLX array to numpy
                if isinstance(images, mx.array):
                    images = np.array(images.tolist())

                # Handle different image formats
                if images.ndim == 4:
                    # Batch of images (N, H, W, C) or (N, C, H, W)
                    if images.shape[-1] in [1, 3, 4]:
                        # (N, H, W, C) -> (N, C, H, W)
                        images = np.transpose(images, (0, 3, 1, 2))

                    # Log first few images
                    n_images = min(8, images.shape[0])
                    self.writer.add_images(
                        f"images/{key}",
                        images[:n_images],
                        epoch,
                        dataformats="NCHW"
                    )
                elif images.ndim == 3:
                    # Single image (H, W, C) or (C, H, W)
                    if images.shape[-1] in [1, 3, 4]:
                        images = np.transpose(images, (2, 0, 1))
                    self.writer.add_image(f"images/{key}", images, epoch)

            except Exception as e:
                print(f"FastMLX-TensorBoard: Error logging image '{key}': {e}")


class TensorBoardEmbedding(Trace):
    """Log embeddings for visualization in TensorBoard projector.

    This trace logs high-dimensional embeddings that can be visualized
    using TensorBoard's embedding projector.

    Args:
        log_dir: Directory to save TensorBoard event files.
        embedding_key: Key in batch containing embeddings to log.
        label_key: Optional key containing labels for coloring.
        metadata_keys: Additional keys to include as metadata.
        log_freq: How often to log embeddings (in epochs).
        max_samples: Maximum number of samples to log.

    Example:
        >>> embedder = TensorBoardEmbedding(
        ...     log_dir="runs/embeddings",
        ...     embedding_key="features",
        ...     label_key="y",
        ...     log_freq=10
        ... )
    """

    def __init__(
        self,
        log_dir: str = "runs",
        embedding_key: str = "embedding",
        label_key: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        log_freq: int = 10,
        max_samples: int = 1000
    ) -> None:
        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard is not installed. Install it with: pip install tensorboard"
            )

        self.log_dir = log_dir
        self.embedding_key = embedding_key
        self.label_key = label_key
        self.metadata_keys = metadata_keys or []
        self.log_freq = log_freq
        self.max_samples = max_samples

        self.writer: Optional[SummaryWriter] = None
        self.embeddings: List = []
        self.labels: List = []
        self.metadata: Dict[str, List] = {k: [] for k in self.metadata_keys}

    def on_start(self, state: MutableMapping[str, object]) -> None:
        """Initialize TensorBoard writer."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_batch_end(
        self,
        batch: MutableMapping[str, object],
        state: MutableMapping[str, object]
    ) -> None:
        """Collect embeddings from batch."""
        if state.get("mode") != "eval":
            return

        embedding = batch.get(self.embedding_key)
        if embedding is None:
            return

        import mlx.core as mx
        import numpy as np

        # Convert to numpy
        if isinstance(embedding, mx.array):
            embedding = np.array(embedding.tolist())

        # Collect embeddings (up to max_samples)
        current_count = len(self.embeddings)
        remaining = self.max_samples - current_count

        if remaining <= 0:
            return

        n_samples = min(embedding.shape[0], remaining)
        self.embeddings.extend(embedding[:n_samples].tolist())

        # Collect labels
        if self.label_key:
            labels = batch.get(self.label_key)
            if labels is not None:
                if isinstance(labels, mx.array):
                    labels = np.array(labels.tolist())
                self.labels.extend(labels[:n_samples].tolist())

        # Collect metadata
        for key in self.metadata_keys:
            data = batch.get(key)
            if data is not None:
                if isinstance(data, mx.array):
                    data = np.array(data.tolist())
                self.metadata[key].extend(data[:n_samples].tolist())

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        """Log collected embeddings."""
        if self.writer is None:
            return

        epoch = state.get("epoch", 0)
        mode = state.get("mode", "train")

        if mode != "eval":
            return

        if (epoch + 1) % self.log_freq != 0:
            return

        if not self.embeddings:
            return

        import numpy as np

        embeddings = np.array(self.embeddings)
        metadata = None

        if self.labels:
            metadata = [str(l) for l in self.labels]

        try:
            self.writer.add_embedding(
                embeddings,
                metadata=metadata,
                global_step=epoch,
                tag=f"embeddings/{self.embedding_key}"
            )
            self.writer.flush()
        except Exception as e:
            print(f"FastMLX-TensorBoard: Error logging embeddings: {e}")

        # Reset collections
        self.embeddings = []
        self.labels = []
        self.metadata = {k: [] for k in self.metadata_keys}

    def on_finish(self, state: MutableMapping[str, object]) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
