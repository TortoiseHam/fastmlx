"""Time series dataset classes for sequential data modeling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx


class TimeSeriesDataset:
    """Dataset for univariate or multivariate time series forecasting.

    Creates sliding window samples from time series data for
    sequence-to-sequence or sequence-to-value prediction.

    Args:
        data: Time series data as list, numpy array, or path to CSV.
               For multivariate, shape should be (time_steps, features).
        input_length: Number of time steps in input window.
        output_length: Number of time steps to predict. If 1, predicts single value.
        stride: Stride between consecutive windows. Default is 1.
        target_column: Column index for target (multivariate only). If None, uses all.
        normalize: Whether to normalize data. Options: None, 'minmax', 'standard'.
        train_ratio: Ratio of data to use for training (for train/val split).

    Example:
        >>> # Univariate forecasting
        >>> data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        >>> dataset = TimeSeriesDataset(data, input_length=3, output_length=1)
        >>> sample = dataset[0]
        >>> print(sample["x"])  # [1, 2, 3]
        >>> print(sample["y"])  # [4]

        >>> # Multivariate forecasting
        >>> data = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]
        >>> dataset = TimeSeriesDataset(data, input_length=2, output_length=1, target_column=0)
    """

    def __init__(
        self,
        data: Union[List, str],
        input_length: int,
        output_length: int = 1,
        stride: int = 1,
        target_column: Optional[int] = None,
        normalize: Optional[str] = None,
        train_ratio: float = 1.0
    ) -> None:
        # Load data from file if path
        if isinstance(data, str):
            data = self._load_csv(data)

        # Convert to list of lists for consistent handling
        if isinstance(data[0], (int, float)):
            # Univariate - convert to 2D
            data = [[x] for x in data]

        self.raw_data = data
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride
        self.target_column = target_column
        self.normalize_method = normalize
        self.train_ratio = train_ratio

        # Number of features
        self.n_features = len(data[0])

        # Normalize data
        self.data, self.normalization_params = self._normalize(data, normalize)

        # Calculate number of samples
        total_len = len(self.data)
        window_size = input_length + output_length
        self._num_samples = max(0, (total_len - window_size) // stride + 1)

    def _load_csv(self, path: str) -> List[List[float]]:
        """Load data from CSV file."""
        import csv

        data = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # Skip header if present

            for row in reader:
                try:
                    data.append([float(x) for x in row])
                except ValueError:
                    continue  # Skip rows that can't be converted

        if not data:
            raise ValueError(f"No valid data found in {path}")

        return data

    def _normalize(
        self,
        data: List[List[float]],
        method: Optional[str]
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """Normalize the data."""
        if method is None:
            return data, {}

        # Convert to column-wise for easier processing
        n_cols = len(data[0])
        columns = [[row[i] for row in data] for i in range(n_cols)]

        params = {"method": method, "columns": {}}

        if method == "minmax":
            normalized_columns = []
            for i, col in enumerate(columns):
                min_val = min(col)
                max_val = max(col)
                range_val = max_val - min_val if max_val != min_val else 1.0
                normalized = [(x - min_val) / range_val for x in col]
                normalized_columns.append(normalized)
                params["columns"][i] = {"min": min_val, "max": max_val}

        elif method == "standard":
            normalized_columns = []
            for i, col in enumerate(columns):
                mean_val = sum(col) / len(col)
                variance = sum((x - mean_val) ** 2 for x in col) / len(col)
                std_val = variance ** 0.5 if variance > 0 else 1.0
                normalized = [(x - mean_val) / std_val for x in col]
                normalized_columns.append(normalized)
                params["columns"][i] = {"mean": mean_val, "std": std_val}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Convert back to row-wise
        normalized_data = [
            [normalized_columns[j][i] for j in range(n_cols)]
            for i in range(len(data))
        ]

        return normalized_data, params

    def denormalize(self, data: mx.array, column: Optional[int] = None) -> mx.array:
        """Denormalize data back to original scale."""
        if not self.normalization_params:
            return data

        method = self.normalization_params["method"]
        col_params = self.normalization_params["columns"]

        if column is not None:
            params = col_params.get(column, {})
            if method == "minmax":
                min_val = params.get("min", 0)
                max_val = params.get("max", 1)
                return data * (max_val - min_val) + min_val
            elif method == "standard":
                mean_val = params.get("mean", 0)
                std_val = params.get("std", 1)
                return data * std_val + mean_val

        return data

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        start = idx * self.stride
        input_end = start + self.input_length
        output_end = input_end + self.output_length

        # Input sequence
        input_seq = self.data[start:input_end]

        # Output sequence
        output_seq = self.data[input_end:output_end]

        # Convert to arrays
        x = mx.array(input_seq)

        if self.target_column is not None:
            y = mx.array([[row[self.target_column]] for row in output_seq])
        else:
            y = mx.array(output_seq)

        # Squeeze if single output
        if self.output_length == 1:
            y = y.squeeze(0)

        return {"x": x, "y": y}


class WindowedDataset:
    """Generic windowed dataset for any sequential data.

    Creates fixed-size windows from sequential data with optional labels.

    Args:
        sequences: List of sequences (each sequence is a list of values).
        labels: Optional labels for each sequence.
        window_size: Size of each window.
        stride: Stride between windows.
        flatten: Whether to flatten windows.

    Example:
        >>> sequences = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        >>> labels = [0, 1]
        >>> dataset = WindowedDataset(sequences, labels, window_size=3)
    """

    def __init__(
        self,
        sequences: List[List[Any]],
        labels: Optional[List[int]] = None,
        window_size: int = 10,
        stride: int = 1,
        flatten: bool = False
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.flatten = flatten

        # Create windows from all sequences
        self.windows: List[List[Any]] = []
        self.window_labels: List[int] = []

        for seq_idx, seq in enumerate(sequences):
            n_windows = max(0, (len(seq) - window_size) // stride + 1)
            for w_idx in range(n_windows):
                start = w_idx * stride
                window = seq[start: start + window_size]
                self.windows.append(window)

                if labels is not None:
                    self.window_labels.append(labels[seq_idx])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        window = self.windows[idx]

        if self.flatten:
            x = mx.array(window).flatten()
        else:
            x = mx.array(window)

        result = {"x": x}

        if self.window_labels:
            result["y"] = mx.array(self.window_labels[idx])

        return result


class OHLCVDataset:
    """Dataset for financial OHLCV (Open, High, Low, Close, Volume) data.

    Specialized dataset for stock/crypto price data with built-in
    feature engineering and technical indicators.

    Args:
        data: OHLCV data as list of dicts with keys 'open', 'high', 'low', 'close', 'volume'.
              Or path to CSV file with these columns.
        input_length: Number of time steps in input window.
        output_length: Number of time steps to predict.
        target: What to predict: 'close', 'return', 'direction', 'high_low'.
        features: List of features to include. Options:
                  'ohlcv' (default), 'returns', 'volatility', 'ma', 'rsi'.
        normalize: Whether to normalize features.

    Example:
        >>> data = [
        ...     {"open": 100, "high": 105, "low": 98, "close": 103, "volume": 1000},
        ...     {"open": 103, "high": 108, "low": 102, "close": 107, "volume": 1200},
        ...     ...
        ... ]
        >>> dataset = OHLCVDataset(data, input_length=30, target='direction')
    """

    def __init__(
        self,
        data: Union[List[Dict[str, float]], str],
        input_length: int = 30,
        output_length: int = 1,
        target: str = "close",
        features: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> None:
        # Load from CSV if path
        if isinstance(data, str):
            data = self._load_ohlcv_csv(data)

        self.raw_data = data
        self.input_length = input_length
        self.output_length = output_length
        self.target = target
        self.features = features or ["ohlcv"]
        self.normalize = normalize

        # Extract raw OHLCV
        self.opens = [d["open"] for d in data]
        self.highs = [d["high"] for d in data]
        self.lows = [d["low"] for d in data]
        self.closes = [d["close"] for d in data]
        self.volumes = [d.get("volume", 0) for d in data]

        # Build feature matrix
        self.feature_matrix = self._build_features()

        # Build targets
        self.targets = self._build_targets()

        # Calculate number of samples
        total_len = len(self.feature_matrix)
        window_size = input_length + output_length
        self._num_samples = max(0, total_len - window_size + 1)

    def _load_ohlcv_csv(self, path: str) -> List[Dict[str, float]]:
        """Load OHLCV data from CSV."""
        import csv

        data = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "open": float(row.get("open", row.get("Open", 0))),
                    "high": float(row.get("high", row.get("High", 0))),
                    "low": float(row.get("low", row.get("Low", 0))),
                    "close": float(row.get("close", row.get("Close", 0))),
                    "volume": float(row.get("volume", row.get("Volume", 0))),
                })

        return data

    def _build_features(self) -> List[List[float]]:
        """Build feature matrix from OHLCV data."""
        n = len(self.closes)
        feature_matrix = []

        for i in range(n):
            features = []

            if "ohlcv" in self.features:
                features.extend([
                    self.opens[i],
                    self.highs[i],
                    self.lows[i],
                    self.closes[i],
                    self.volumes[i],
                ])

            if "returns" in self.features and i > 0:
                # Safe division for returns
                prev_close = self.closes[i - 1]
                if prev_close != 0:
                    ret = (self.closes[i] - prev_close) / prev_close
                else:
                    ret = 0.0
                features.append(ret)
            elif "returns" in self.features:
                features.append(0.0)

            if "volatility" in self.features:
                # Simple volatility: (high - low) / close, with safe division
                if self.closes[i] != 0:
                    vol = (self.highs[i] - self.lows[i]) / self.closes[i]
                else:
                    vol = 0.0
                features.append(vol)

            if "ma" in self.features:
                # Moving averages (5, 10, 20 periods)
                for period in [5, 10, 20]:
                    if i >= period - 1:
                        ma = sum(self.closes[i - period + 1: i + 1]) / period
                    else:
                        ma = self.closes[i]
                    # Safe division for MA ratio
                    if self.closes[i] != 0:
                        features.append(ma / self.closes[i] - 1)
                    else:
                        features.append(0.0)

            feature_matrix.append(features)

        # Normalize if requested
        if self.normalize and feature_matrix:
            feature_matrix = self._normalize_features(feature_matrix)

        return feature_matrix

    def _normalize_features(
        self,
        features: List[List[float]]
    ) -> List[List[float]]:
        """Normalize features using min-max scaling."""
        if not features:
            return features

        n_features = len(features[0])
        normalized = []

        # Compute min/max for each feature
        mins = [min(row[i] for row in features) for i in range(n_features)]
        maxs = [max(row[i] for row in features) for i in range(n_features)]

        for row in features:
            norm_row = []
            for i, val in enumerate(row):
                range_val = maxs[i] - mins[i]
                if range_val > 0:
                    norm_row.append((val - mins[i]) / range_val)
                else:
                    norm_row.append(0.0)
            normalized.append(norm_row)

        return normalized

    def _build_targets(self) -> List[Any]:
        """Build target values based on target type."""
        n = len(self.closes)
        targets = []

        for i in range(n):
            if i + self.output_length >= n:
                targets.append(None)
                continue

            future_close = self.closes[i + self.output_length]

            if self.target == "close":
                targets.append(future_close)

            elif self.target == "return":
                ret = (future_close - self.closes[i]) / self.closes[i]
                targets.append(ret)

            elif self.target == "direction":
                direction = 1 if future_close > self.closes[i] else 0
                targets.append(direction)

            elif self.target == "high_low":
                future_high = max(self.highs[i + 1: i + self.output_length + 1])
                future_low = min(self.lows[i + 1: i + self.output_length + 1])
                targets.append([future_high, future_low])

            else:
                targets.append(future_close)

        return targets

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        input_end = idx + self.input_length

        # Input features
        x = mx.array(self.feature_matrix[idx:input_end])

        # Target
        target = self.targets[input_end - 1]
        if isinstance(target, list):
            y = mx.array(target)
        else:
            y = mx.array([target])

        return {"x": x, "y": y}
