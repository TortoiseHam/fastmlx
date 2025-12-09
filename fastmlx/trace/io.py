"""Input/output related traces for model saving and logging."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from .base import Trace


class BestModelSaver(Trace):
    """Save model weights when a monitored metric improves.

    Args:
        model: The model to save.
        save_dir: Directory to save the model.
        metric: Metric name to monitor (from state['metrics']).
        save_best_mode: One of 'max' or 'min'. Save when metric is highest or lowest.
        mode: Mode(s) in which to run. Defaults to "eval".
    """

    def __init__(
        self,
        model,
        save_dir: str,
        metric: str = "accuracy",
        save_best_mode: str = "max",
        mode: Optional[Union[str, List[str]]] = "eval",
    ) -> None:
        super().__init__(inputs=[metric], mode=mode)
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.metric = metric
        self.save_best_mode = save_best_mode
        self.best: Optional[float] = None

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        score = state['metrics'].get(self.metric)
        if score is None:
            return
        if self.best is None or (self.save_best_mode == "max" and score > self.best) or (
            self.save_best_mode == "min" and score < self.best
        ):
            self.best = score
            path = os.path.join(self.save_dir, "model_best.npz")
            self.model.save_weights(path)
            print(f"FastMLX-BestModelSaver: Saved model to {path}")


class ModelSaver(Trace):
    """Save model weights at specified intervals.

    Args:
        model: The model to save.
        save_dir: Directory to save the model.
        frequency: How often to save, in epochs. If 0, only save at the end.
        save_optimizer: Whether to save optimizer state as well.
        mode: Mode(s) in which to run. Defaults to "train".
    """

    def __init__(
        self,
        model,
        save_dir: str,
        frequency: int = 1,
        save_optimizer: bool = False,
        mode: Optional[Union[str, List[str]]] = "train",
    ) -> None:
        super().__init__(mode=mode)
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.frequency = frequency
        self.save_optimizer = save_optimizer

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        epoch = state.get('epoch', 0)
        if self.frequency > 0 and (epoch + 1) % self.frequency == 0:
            self._save(epoch)

    def on_finish(self, state: MutableMapping[str, object]) -> None:
        epoch = state.get('epoch', 0)
        self._save(epoch, final=True)

    def _save(self, epoch: int, final: bool = False) -> None:
        if final:
            model_path = os.path.join(self.save_dir, "model_final.npz")
        else:
            model_path = os.path.join(self.save_dir, f"model_epoch_{epoch + 1}.npz")

        self.model.save_weights(model_path)
        print(f"FastMLX-ModelSaver: Saved model to {model_path}")

        if self.save_optimizer and hasattr(self.model, 'optimizer'):
            if final:
                opt_path = os.path.join(self.save_dir, "optimizer_final.npz")
            else:
                opt_path = os.path.join(self.save_dir, f"optimizer_epoch_{epoch + 1}.npz")
            # Save optimizer state
            opt_state = self.model.optimizer.state
            mx.savez(opt_path, **{f"opt_{k}": v for k, v in enumerate(opt_state) if isinstance(v, mx.array)})


class CSVLogger(Trace):
    """Log training metrics to a CSV file.

    Args:
        filename: Path to the CSV file.
        separator: Delimiter for the CSV file.
        append: If True, append to existing file. If False, overwrite.
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        filename: str,
        separator: str = ",",
        append: bool = False,
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(mode=mode)
        self.filename = filename
        self.separator = separator
        self.append = append
        self.keys: Optional[List[str]] = None
        self.file_handle = None
        self.writer = None

    def on_start(self, state: MutableMapping[str, object]) -> None:
        mode = 'a' if self.append else 'w'
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filename) if os.path.dirname(self.filename) else '.', exist_ok=True)
        self.file_handle = open(self.filename, mode, newline='')
        self.writer = csv.writer(self.file_handle, delimiter=self.separator)

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        metrics = state.get('metrics', {})
        epoch = state.get('epoch', 0)
        mode = state.get('mode', 'train')

        row_dict: Dict[str, Any] = {'epoch': epoch, 'mode': mode}
        row_dict.update(metrics)

        # Write header on first epoch
        if self.keys is None:
            self.keys = list(row_dict.keys())
            if not self.append:
                self.writer.writerow(self.keys)

        # Write values
        row = [row_dict.get(k, '') for k in self.keys]
        self.writer.writerow(row)
        self.file_handle.flush()

    def on_finish(self, state: MutableMapping[str, object]) -> None:
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None


class ProgressLogger(Trace):
    """Print training progress to console.

    Args:
        log_frequency: How often to log (in batches). If 0, log only at epoch end.
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        log_frequency: int = 0,
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(mode=mode)
        self.log_frequency = log_frequency
        self.batch_count: int = 0

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.batch_count = 0

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        self.batch_count += 1
        if self.log_frequency > 0 and self.batch_count % self.log_frequency == 0:
            epoch = state.get('epoch', 0)
            mode = state.get('mode', 'train')
            loss = batch.get('loss') or batch.get('ce')
            if loss is not None:
                if hasattr(loss, 'item'):
                    loss = float(loss.item())
                print(f"Epoch {epoch} [{mode}] Batch {self.batch_count}: loss={loss:.4f}")


class Timer(Trace):
    """Track and report training time.

    Records time for each epoch and total training time.

    Args:
        mode: Mode(s) in which to run. Defaults to None (all modes).
    """

    def __init__(
        self,
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(outputs=["epoch_time"], mode=mode)
        self.start_time: Optional[datetime] = None
        self.epoch_start: Optional[datetime] = None

    def on_start(self, state: MutableMapping[str, object]) -> None:
        self.start_time = datetime.now()
        print(f"FastMLX-Timer: Training started at {self.start_time}")

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.epoch_start = datetime.now()

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        if self.epoch_start:
            epoch_time = datetime.now() - self.epoch_start
            epoch = state.get('epoch', 0)
            mode = state.get('mode', 'train')
            state['metrics']['epoch_time'] = epoch_time.total_seconds()
            print(f"FastMLX-Timer: Epoch {epoch} [{mode}] completed in {epoch_time}")

    def on_finish(self, state: MutableMapping[str, object]) -> None:
        if self.start_time:
            total_time = datetime.now() - self.start_time
            print(f"FastMLX-Timer: Training completed in {total_time}")


class ImageSaver(Trace):
    """Save images to disk during training or evaluation.

    Can save raw images, augmented batches, generated outputs, or any image
    data stored in the batch dictionary.

    Args:
        inputs: Key(s) for images to save from batch.
        save_dir: Directory to save images.
        prefix: Prefix for saved filenames.
        format: Image format ('png', 'jpg', 'npy').
        frequency: How often to save (in epochs). 0 means save every epoch.
        max_images: Maximum number of images to save per epoch.
        denormalize: Function to denormalize images before saving.
        mode: Mode(s) in which to run. Defaults to ["eval", "test"].

    Example:
        >>> ImageSaver(inputs="x_aug", save_dir="./augmented_images")
        >>> ImageSaver(inputs=["x", "x_reconstructed"], save_dir="./outputs")
    """

    def __init__(
        self,
        inputs: Union[str, Sequence[str]],
        save_dir: str,
        prefix: str = "image",
        format: str = "png",
        frequency: int = 1,
        max_images: int = 16,
        denormalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        if isinstance(inputs, str):
            inputs = [inputs]
        super().__init__(inputs=list(inputs), mode=mode if mode else ["eval", "test"])
        self.input_keys = list(inputs)
        self.save_dir = save_dir
        self.prefix = prefix
        self.format = format.lower()
        self.frequency = frequency
        self.max_images = max_images
        self.denormalize = denormalize
        self.images_saved: int = 0
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.images_saved = 0

    def _save_image(self, img: np.ndarray, path: str) -> None:
        """Save a single image to disk."""
        # Denormalize if needed
        if self.denormalize is not None:
            img = self.denormalize(img)

        # Handle different image formats
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Assume normalized to [0, 1] or [-1, 1]
            if img.min() < 0:
                img = (img + 1) / 2  # [-1, 1] -> [0, 1]
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Handle grayscale
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        # Handle single channel
        if img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)

        if self.format == 'npy':
            np.save(path, img)
        else:
            # Try to use PIL for standard formats
            try:
                from PIL import Image
                if img.ndim == 2:
                    pil_img = Image.fromarray(img, mode='L')
                else:
                    pil_img = Image.fromarray(img, mode='RGB')
                pil_img.save(path)
            except ImportError:
                # Fallback to numpy
                np.save(path.replace(f'.{self.format}', '.npy'), img)
                print(f"Warning: PIL not available, saved as .npy instead")

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        if self.images_saved >= self.max_images:
            return

        epoch = state.get('epoch', 0)
        mode = state.get('mode', 'eval')

        if self.frequency > 0 and (epoch + 1) % self.frequency != 0:
            return

        for key in self.input_keys:
            images = batch.get(key)
            if images is None:
                continue

            # Convert to numpy
            if isinstance(images, mx.array):
                images = np.array(images)

            # Handle batch vs single image
            if images.ndim == 3:
                images = np.expand_dims(images, axis=0)

            # Save each image
            for i, img in enumerate(images):
                if self.images_saved >= self.max_images:
                    break

                filename = f"{self.prefix}_{key}_epoch{epoch}_{mode}_{self.images_saved}.{self.format}"
                path = os.path.join(self.save_dir, filename)
                self._save_image(img, path)
                self.images_saved += 1
                print(f"FastMLX-ImageSaver: Saved {path}")


class ImageViewer(Trace):
    """Display images during training for real-time visualization.

    Shows images in a grid format at specified intervals. Useful for
    monitoring augmentations, generated outputs, or reconstructions.

    Args:
        inputs: Key(s) for images to display from batch.
        n_images: Number of images to display.
        frequency: How often to display (in batches). 0 means display at epoch end only.
        title: Title for the display.
        mode: Mode(s) in which to run. Defaults to ["eval"].

    Note:
        Requires matplotlib to be installed.

    Example:
        >>> ImageViewer(inputs="x_generated", n_images=8, frequency=100)
    """

    def __init__(
        self,
        inputs: Union[str, Sequence[str]],
        n_images: int = 8,
        frequency: int = 0,
        title: str = "Images",
        mode: Optional[Union[str, List[str]]] = None,
    ) -> None:
        if isinstance(inputs, str):
            inputs = [inputs]
        super().__init__(inputs=list(inputs), mode=mode if mode else ["eval"])
        self.input_keys = list(inputs)
        self.n_images = n_images
        self.frequency = frequency
        self.title = title
        self.batch_count: int = 0
        self.collected_images: Dict[str, List[np.ndarray]] = {}

    def on_epoch_begin(self, state: MutableMapping[str, object]) -> None:
        self.batch_count = 0
        self.collected_images = {key: [] for key in self.input_keys}

    def _display_images(self, state: MutableMapping[str, object]) -> None:
        """Display collected images in a grid."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available for ImageViewer")
            return

        epoch = state.get('epoch', 0)
        mode = state.get('mode', 'eval')

        n_keys = len(self.input_keys)
        n_cols = min(self.n_images, 8)

        for key in self.input_keys:
            images = self.collected_images.get(key, [])
            if not images:
                continue

            n_show = min(len(images), self.n_images)
            n_rows = (n_show + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_rows == 1:
                axes = [axes]
            elif n_cols == 1:
                axes = [[ax] for ax in axes]

            fig.suptitle(f"{self.title} - {key} (Epoch {epoch}, {mode})")

            for i in range(n_rows):
                for j in range(n_cols):
                    idx = i * n_cols + j
                    ax = axes[i][j]
                    if idx < n_show:
                        img = images[idx]
                        if img.ndim == 2:
                            ax.imshow(img, cmap='gray')
                        else:
                            # Normalize if needed
                            if img.min() < 0:
                                img = (img + 1) / 2
                            img = np.clip(img, 0, 1)
                            ax.imshow(img)
                    ax.axis('off')

            plt.tight_layout()
            plt.show()

    def on_batch_end(self, batch: MutableMapping[str, object], state: MutableMapping[str, object]) -> None:
        self.batch_count += 1

        # Collect images
        for key in self.input_keys:
            images = batch.get(key)
            if images is None:
                continue

            if isinstance(images, mx.array):
                images = np.array(images)

            if images.ndim == 3:
                images = np.expand_dims(images, axis=0)

            n_to_add = min(images.shape[0], self.n_images - len(self.collected_images[key]))
            for i in range(n_to_add):
                self.collected_images[key].append(images[i])

        # Display at frequency intervals
        if self.frequency > 0 and self.batch_count % self.frequency == 0:
            self._display_images(state)
            # Reset collection
            self.collected_images = {key: [] for key in self.input_keys}

    def on_epoch_end(self, state: MutableMapping[str, object]) -> None:
        # Display at epoch end if not done via frequency
        if self.frequency == 0:
            self._display_images(state)
