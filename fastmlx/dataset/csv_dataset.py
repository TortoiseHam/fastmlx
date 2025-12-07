"""CSV Dataset implementation."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from .mlx_dataset import MLXDataset


class CSVDataset:
    """Dataset for loading data from CSV files.

    Args:
        file_path: Path to the CSV file.
        columns: List of column names to load. If None, loads all columns.
        delimiter: CSV delimiter character.
        skip_header: Whether to skip the first row (header).
        feature_columns: Columns to use as features (stored as 'x').
        label_column: Column to use as label (stored as 'y').
        dtype: Data type for numeric conversions.

    Example:
        >>> dataset = CSVDataset(
        ...     "data.csv",
        ...     feature_columns=["col1", "col2", "col3"],
        ...     label_column="target"
        ... )
        >>> print(len(dataset))
        1000
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['x', 'y'])
    """

    def __init__(
        self,
        file_path: str,
        columns: Optional[List[str]] = None,
        delimiter: str = ",",
        skip_header: bool = True,
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
        dtype: mx.Dtype = mx.float32
    ) -> None:
        self.file_path = file_path
        self.delimiter = delimiter
        self.dtype = dtype
        self.feature_columns = feature_columns
        self.label_column = label_column

        # Load CSV
        self.data: Dict[str, List[Any]] = {}
        self.header: List[str] = []

        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)

            # Read header
            if skip_header:
                self.header = next(reader)
            else:
                # Use column indices as names
                first_row = next(reader)
                self.header = [f"col_{i}" for i in range(len(first_row))]
                # Process first row as data
                for i, val in enumerate(first_row):
                    col_name = self.header[i]
                    if columns is None or col_name in columns:
                        if col_name not in self.data:
                            self.data[col_name] = []
                        self.data[col_name].append(self._parse_value(val))

            # Filter columns
            if columns is not None:
                self.header = [h for h in self.header if h in columns]

            # Initialize data dict
            for col in self.header:
                if col not in self.data:
                    self.data[col] = []

            # Read data rows
            for row in reader:
                for i, col_name in enumerate(self.header):
                    if i < len(row):
                        self.data[col_name].append(self._parse_value(row[i]))

        self.size = len(self.data[self.header[0]]) if self.header else 0

    def _parse_value(self, val: str) -> Union[float, str]:
        """Parse a string value to float if possible."""
        try:
            return float(val)
        except ValueError:
            return val

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample by index.

        Returns numpy arrays to avoid Metal buffer allocation limits.
        Conversion to MLX arrays happens at batch time.
        """
        # Map MLX dtypes to numpy dtypes
        np_dtype = np.float32
        if self.dtype == mx.float64:
            np_dtype = np.float64
        elif self.dtype == mx.float16:
            np_dtype = np.float16

        if self.feature_columns and self.label_column:
            # Return structured x, y format
            features = [self.data[col][idx] for col in self.feature_columns]
            label = self.data[self.label_column][idx]
            return {
                "x": np.array(features, dtype=np_dtype),
                "y": np.array([label], dtype=np_dtype if isinstance(label, float) else np.int32)
            }
        else:
            # Return all columns
            return {
                col: np.array([self.data[col][idx]], dtype=np_dtype)
                for col in self.header
            }

    def to_mlx_dataset(self) -> "MLXDataset":
        """Convert to MLXDataset for in-memory access."""
        from .mlx_dataset import MLXDataset

        if self.feature_columns and self.label_column:
            features = [[self.data[col][i] for col in self.feature_columns]
                       for i in range(self.size)]
            labels = [self.data[self.label_column][i] for i in range(self.size)]
            return MLXDataset({
                "x": mx.array(features, dtype=self.dtype),
                "y": mx.array(labels, dtype=mx.int32 if all(isinstance(label, int) for label in labels) else self.dtype)
            })
        else:
            data = {
                col: mx.array(self.data[col], dtype=self.dtype)
                for col in self.header
            }
            return MLXDataset(data)
