"""Utilities for loading the ciFAIR10 dataset using MLX."""

from __future__ import annotations

import os
import pickle
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import mlx.core as mx

from ..mlx_dataset import MLXDataset

_CIFAIR10_FILE_ID = "1dqTgqMVvgx_FZNAC7TqzoA0hYX1ttOUq"


def _download_file_from_google_drive(file_id: str, destination: str) -> None:
    """Download a file from Google Drive if it does not already exist."""

    if os.path.exists(destination):
        return

    base_url = "https://drive.google.com/uc?export=download"
    initial_req = urllib.request.Request(
        f"{base_url}&id={file_id}", headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(initial_req) as response:
        if response.getheader("Content-Type", "").startswith("text/html"):
            html = response.read().decode("utf-8")
            match = re.search(r"confirm=([0-9A-Za-z_]+)", html)
            if not match:
                raise RuntimeError("Unable to obtain download confirmation token")
            confirm_token = match.group(1)
            confirm_req = urllib.request.Request(
                f"{base_url}&id={file_id}&confirm={confirm_token}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(confirm_req) as confirm_response, open(
                destination, "wb"
            ) as out_file:
                shutil.copyfileobj(confirm_response, out_file)
        else:
            with open(destination, "wb") as out_file:
                shutil.copyfileobj(response, out_file)


def _load_batch(file_path: str, label_key: str = "labels") -> Tuple[np.ndarray, np.ndarray]:
    """Load a single ciFAIR batch."""
    with open(file_path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        d = {k.decode("utf-8") if isinstance(k, bytes) else k: v for k, v in d.items()}
    data = d["data"].reshape(-1, 3, 32, 32)
    labels = np.array(d[label_key], dtype=np.uint8)
    return data, labels


def load_data(
    root_dir: str | None = None,
    image_key: str = "x",
    label_key: str = "y",
) -> Tuple[MLXDataset, MLXDataset]:
    """Load the ciFAIR10 dataset.

    Args:
        root_dir: Directory to store the downloaded data. Defaults to ``~/fastmlx_data/ciFAIR10``.
        image_key: Key name for images in the returned datasets.
        label_key: Key name for labels in the returned datasets.

    Returns:
        A tuple of training and test datasets.
    """

    home = str(Path.home())
    if root_dir is None:
        root_dir = os.path.join(home, "fastmlx_data", "ciFAIR10")
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), "ciFAIR10")
    os.makedirs(root_dir, exist_ok=True)

    compressed_path = os.path.join(root_dir, "ciFAIR10.zip")
    extracted_path = os.path.join(root_dir, "ciFAIR-10")

    if not os.path.exists(extracted_path):
        print(f"Downloading data to {root_dir}")
        _download_file_from_google_drive(_CIFAIR10_FILE_ID, compressed_path)
        if not zipfile.is_zipfile(compressed_path):
            raise RuntimeError(f"Downloaded file {compressed_path} is not a valid zip archive")
        print(f"Extracting data to {root_dir}")
        shutil.unpack_archive(compressed_path, root_dir)

    num_train_samples = 50000
    x_train = np.empty((num_train_samples, 3, 32, 32), dtype=np.uint8)
    y_train = np.empty((num_train_samples,), dtype=np.uint8)

    for i in range(1, 6):
        fpath = os.path.join(extracted_path, f"data_batch_{i}")
        data, labels = _load_batch(fpath)
        x_train[(i - 1) * 10000 : i * 10000] = data
        y_train[(i - 1) * 10000 : i * 10000] = labels

    fpath = os.path.join(extracted_path, "test_batch")
    x_test, y_test = _load_batch(fpath)

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    train = MLXDataset({image_key: mx.array(x_train), label_key: mx.array(y_train)})
    test = MLXDataset({image_key: mx.array(x_test), label_key: mx.array(y_test)})
    return train, test
