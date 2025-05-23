import os
import mlx.core as mx


class BestModelSaver:
    def __init__(self, model, save_dir, metric="accuracy", save_best_mode="max"):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.metric = metric
        self.save_best_mode = save_best_mode
        self.best = None

    def on_epoch_end(self, state):
        score = state['metrics'].get(self.metric)
        if score is None:
            return
        if self.best is None or (self.save_best_mode == "max" and score > self.best) or (
            self.save_best_mode == "min" and score < self.best
        ):
            self.best = score
            path = os.path.join(self.save_dir, "model.npz")
            mx.savez(path, *self.model.parameters())
