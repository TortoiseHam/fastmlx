"""Input/output related traces."""

from __future__ import annotations

import os
from typing import MutableMapping, Optional

import mlx.core as mx


class BestModelSaver:
    def __init__(self, model, save_dir: str, metric: str = "accuracy", save_best_mode: str = "max") -> None:
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
            path = os.path.join(self.save_dir, "model.npz")
            # Utilize MLX's built-in model saving utility to properly handle
            # nested parameter structures.
            self.model.save_weights(path)
