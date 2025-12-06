"""Datasets used by FastMLX."""

from .csv_dataset import CSVDataset
from .dir_dataset import DirDataset, LabeledDirDataset
from .generator_dataset import (
    BatchDataset,
    CombinedDataset,
    GeneratorDataset,
    InterleaveDataset,
)
from .mlx_dataset import MLXDataset

# NLP datasets
from .nlp_dataset import (
    LanguageModelDataset,
    SequenceDataset,
    TextDataset,
    TokenizedDataset,
)

# Time series datasets
from .timeseries_dataset import (
    OHLCVDataset,
    TimeSeriesDataset,
    WindowedDataset,
)

__all__ = [
    # Core datasets
    "MLXDataset",
    "CSVDataset",
    "DirDataset",
    "LabeledDirDataset",
    "GeneratorDataset",
    "BatchDataset",
    "CombinedDataset",
    "InterleaveDataset",
    # NLP datasets
    "TextDataset",
    "SequenceDataset",
    "TokenizedDataset",
    "LanguageModelDataset",
    # Time series datasets
    "TimeSeriesDataset",
    "WindowedDataset",
    "OHLCVDataset",
]
