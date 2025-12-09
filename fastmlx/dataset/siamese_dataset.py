"""Siamese dataset for one-shot and few-shot learning."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import mlx.core as mx
import numpy as np

from .dir_dataset import LabeledDirDataset


class SiameseDirDataset(LabeledDirDataset):
    """Dataset that returns pairs of samples for Siamese network training.

    For each sample, returns a pair of images and a label indicating whether
    they are from the same class (1) or different classes (0).

    Args:
        root_dir: Root directory containing class subdirectories.
        file_extension: File extension filter (e.g., '.png').
        transform: Optional transform function to apply to images.
        class_to_idx: Optional mapping from class name to index.
        percent_same_class: Probability of returning a same-class pair (default 0.5).

    Example:
        >>> dataset = SiameseDirDataset("/path/to/omniglot", file_extension=".png")
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['x_left', 'x_right', 'y'])  # y=1 for same class, y=0 for different
    """

    def __init__(
        self,
        root_dir: str,
        file_extension: Optional[str] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        percent_same_class: float = 0.5
    ) -> None:
        super().__init__(root_dir, file_extension, transform, class_to_idx)
        self.percent_same_class = percent_same_class

        # Build class-to-indices mapping for efficient sampling
        self._class_to_indices: Dict[int, List[int]] = {}
        for idx, (path, class_idx) in enumerate(self.samples):
            if class_idx not in self._class_to_indices:
                self._class_to_indices[class_idx] = []
            self._class_to_indices[class_idx].append(idx)

        # Get list of valid classes (classes with at least 2 samples for same-class pairs)
        self._valid_classes = [c for c, indices in self._class_to_indices.items() if len(indices) >= 1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get first image (anchor)
        path1, class1 = self.samples[idx]
        img1 = self._load_file(path1)

        # Decide if same class or different class
        if random.random() < self.percent_same_class and len(self._class_to_indices[class1]) > 1:
            # Same class pair
            same_class_indices = [i for i in self._class_to_indices[class1] if i != idx]
            if same_class_indices:
                idx2 = random.choice(same_class_indices)
                path2, _ = self.samples[idx2]
                img2 = self._load_file(path2)
                label = 1  # Same class
            else:
                # Fallback to different class
                other_classes = [c for c in self._valid_classes if c != class1]
                class2 = random.choice(other_classes) if other_classes else class1
                idx2 = random.choice(self._class_to_indices[class2])
                path2, _ = self.samples[idx2]
                img2 = self._load_file(path2)
                label = 0 if class2 != class1 else 1
        else:
            # Different class pair
            other_classes = [c for c in self._valid_classes if c != class1]
            if other_classes:
                class2 = random.choice(other_classes)
                idx2 = random.choice(self._class_to_indices[class2])
                path2, _ = self.samples[idx2]
                img2 = self._load_file(path2)
                label = 0  # Different class
            else:
                # Only one class, return same-class pair
                same_class_indices = [i for i in self._class_to_indices[class1] if i != idx]
                idx2 = random.choice(same_class_indices) if same_class_indices else idx
                path2, _ = self.samples[idx2]
                img2 = self._load_file(path2)
                label = 1

        # Apply transforms
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            "x_left": img1,
            "x_right": img2,
            "y": mx.array([label], dtype=mx.float32)
        }

    def one_shot_trial(self, n_way: int = 20, n_trials: int = 400) -> List[Dict[str, Any]]:
        """Generate one-shot learning trials.

        Creates trials for one-shot accuracy evaluation where the model must
        identify which of n_way images matches the test image.

        Args:
            n_way: Number of classes per trial (default 20 for Omniglot).
            n_trials: Number of trials to generate.

        Returns:
            List of trial dictionaries with 'test_image', 'support_images', and 'correct_idx'.
        """
        trials = []

        for _ in range(n_trials):
            # Sample n_way classes
            if len(self._valid_classes) < n_way:
                sampled_classes = random.choices(self._valid_classes, k=n_way)
            else:
                sampled_classes = random.sample(self._valid_classes, n_way)

            # Pick correct class (first one)
            correct_class = sampled_classes[0]
            correct_idx = 0

            # Get test image from correct class
            test_idx = random.choice(self._class_to_indices[correct_class])
            test_path, _ = self.samples[test_idx]
            test_image = self._load_file(test_path)

            # Get support images (one from each class, different from test for correct class)
            support_images = []
            for class_idx in sampled_classes:
                available_indices = [i for i in self._class_to_indices[class_idx] if i != test_idx]
                if not available_indices:
                    available_indices = self._class_to_indices[class_idx]
                support_idx = random.choice(available_indices)
                support_path, _ = self.samples[support_idx]
                support_images.append(self._load_file(support_path))

            if self.transform is not None:
                test_image = self.transform(test_image)
                support_images = [self.transform(img) for img in support_images]

            trials.append({
                "test_image": test_image,
                "support_images": mx.stack(support_images),
                "correct_idx": correct_idx
            })

        return trials

    def get_n_shot_batch(
        self,
        n_way: int = 5,
        n_shot: int = 1,
        n_query: int = 1
    ) -> Dict[str, Any]:
        """Generate an n-way k-shot learning batch.

        Args:
            n_way: Number of classes per episode.
            n_shot: Number of support samples per class.
            n_query: Number of query samples per class.

        Returns:
            Dictionary with 'support_images', 'support_labels', 'query_images', 'query_labels'.
        """
        # Sample n_way classes
        if len(self._valid_classes) < n_way:
            sampled_classes = random.choices(self._valid_classes, k=n_way)
        else:
            sampled_classes = random.sample(self._valid_classes, n_way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for way_idx, class_idx in enumerate(sampled_classes):
            indices = self._class_to_indices[class_idx]
            n_samples_needed = n_shot + n_query

            if len(indices) >= n_samples_needed:
                selected = random.sample(indices, n_samples_needed)
            else:
                selected = random.choices(indices, k=n_samples_needed)

            # Split into support and query
            support_indices = selected[:n_shot]
            query_indices = selected[n_shot:]

            for idx in support_indices:
                path, _ = self.samples[idx]
                img = self._load_file(path)
                if self.transform:
                    img = self.transform(img)
                support_images.append(img)
                support_labels.append(way_idx)

            for idx in query_indices:
                path, _ = self.samples[idx]
                img = self._load_file(path)
                if self.transform:
                    img = self.transform(img)
                query_images.append(img)
                query_labels.append(way_idx)

        return {
            "support_images": mx.stack(support_images),
            "support_labels": mx.array(support_labels, dtype=mx.int32),
            "query_images": mx.stack(query_images),
            "query_labels": mx.array(query_labels, dtype=mx.int32)
        }
