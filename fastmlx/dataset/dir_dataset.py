"""Directory-based Dataset implementations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np


class DirDataset:
    """Dataset for loading files from a directory.

    Args:
        root_dir: Root directory containing the data files.
        file_extension: File extension filter (e.g., '.png', '.jpg', '.npy').
                       If None, includes all files.
        recursive: Whether to search subdirectories recursively.
        transform: Optional transform function to apply to loaded data.

    Example:
        >>> dataset = DirDataset("/path/to/images", file_extension=".png")
        >>> print(len(dataset))
        1000
    """

    def __init__(
        self,
        root_dir: str,
        file_extension: Optional[str] = None,
        recursive: bool = True,
        transform: Optional[Callable] = None
    ) -> None:
        self.root_dir = Path(root_dir)
        self.file_extension = file_extension
        self.recursive = recursive
        self.transform = transform

        # Collect file paths
        self.file_paths: List[Path] = []
        self._scan_directory()

    def _scan_directory(self) -> None:
        """Scan directory for files."""
        if self.recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for path in self.root_dir.glob(pattern):
            if path.is_file():
                if self.file_extension is None or path.suffix.lower() == self.file_extension.lower():
                    self.file_paths.append(path)

        self.file_paths.sort()

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.file_paths[idx]
        data = self._load_file(path)

        if self.transform is not None:
            data = self.transform(data)

        return {"x": data, "path": str(path)}

    def _load_file(self, path: Path) -> mx.array:
        """Load a file based on its extension."""
        suffix = path.suffix.lower()

        if suffix == ".npy":
            return mx.array(np.load(path))
        elif suffix == ".npz":
            data = np.load(path)
            # Return first array in npz
            return mx.array(data[list(data.keys())[0]])
        elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
            return self._load_image(path)
        else:
            # Try to load as numpy array
            try:
                return mx.array(np.load(path))
            except Exception:
                raise ValueError(f"Unsupported file format: {suffix}")

    def _load_image(self, path: Path) -> mx.array:
        """Load an image file."""
        try:
            from PIL import Image
            img = Image.open(path)
            img_array = np.array(img)
            return mx.array(img_array)
        except ImportError:
            raise ImportError("PIL is required to load images. Install with: pip install Pillow")


class LabeledDirDataset:
    """Dataset for loading labeled data from a directory structure.

    Expects directory structure like:
        root/
            class_0/
                img1.png
                img2.png
            class_1/
                img3.png
                img4.png

    Args:
        root_dir: Root directory containing class subdirectories.
        file_extension: File extension filter.
        transform: Optional transform function.
        class_to_idx: Optional mapping from class name to index.
                     If None, classes are sorted alphabetically.

    Example:
        >>> dataset = LabeledDirDataset("/path/to/labeled_images")
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['x', 'y', 'path'])
    """

    def __init__(
        self,
        root_dir: str,
        file_extension: Optional[str] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ) -> None:
        self.root_dir = Path(root_dir)
        self.file_extension = file_extension
        self.transform = transform

        # Find classes
        self.classes: List[str] = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect samples
        self.samples: List[Tuple[Path, int]] = []
        self._scan_directory()

    def _scan_directory(self) -> None:
        """Scan directory for labeled samples."""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for path in class_dir.iterdir():
                if path.is_file():
                    if self.file_extension is None or path.suffix.lower() == self.file_extension.lower():
                        self.samples.append((path, class_idx))

        # Sort for reproducibility
        self.samples.sort(key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, label = self.samples[idx]
        data = self._load_file(path)

        if self.transform is not None:
            data = self.transform(data)

        return {
            "x": data,
            "y": mx.array([label], dtype=mx.int32),
            "path": str(path)
        }

    def _load_file(self, path: Path) -> mx.array:
        """Load a file based on its extension."""
        suffix = path.suffix.lower()

        if suffix == ".npy":
            return mx.array(np.load(path))
        elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
            return self._load_image(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_image(self, path: Path) -> mx.array:
        """Load an image file."""
        try:
            from PIL import Image
            img = Image.open(path)
            img_array = np.array(img)
            return mx.array(img_array)
        except ImportError:
            raise ImportError("PIL is required to load images. Install with: pip install Pillow")

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.classes)
