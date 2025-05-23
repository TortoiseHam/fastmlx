"""Training and evaluation orchestration."""

from __future__ import annotations

from typing import Iterable, List, MutableMapping, Optional

import mlx.core as mx

from .pipeline import Pipeline
from .network import Network


class Estimator:
    """Run a :class:`~fastmlx.pipeline.Pipeline` using a :class:`~fastmlx.network.Network`."""

    def __init__(self, pipeline: Pipeline, network: Network, epochs: int,
                 traces: Optional[Iterable[object]] | None = None,
                 log_interval: int = 100) -> None:
        self.pipeline: Pipeline = pipeline
        self.network: Network = network
        self.epochs: int = epochs
        self.traces: List[object] = list(traces or [])
        self.log_interval = log_interval

    def fit(self) -> MutableMapping[str, object]:
        """Train the network with periodic logging similar to FastEstimator."""

        import time

        state: MutableMapping[str, object] = {}
        step = 0
        print(
            f"FastMLX-Start: step: 1; logging_interval: {self.log_interval}; num_device: 1;"
        )
        for epoch in range(self.epochs):
            epoch_start = time.time()
            state = {"mode": "train", "epoch": epoch, "metrics": {}}
            for t in self.traces:
                if hasattr(t, "on_epoch_begin"):
                    t.on_epoch_begin(state)
            for batch in self.pipeline.get_loader("train"):
                step += 1
                batch_start = time.time()
                state["batch"] = batch
                self.network.run(batch, state)
                for t in self.traces:
                    if hasattr(t, "on_batch_end"):
                        t.on_batch_end(batch, state)
                if step % self.log_interval == 0:
                    lr = getattr(self.network.ops[0].model.optimizer, "learning_rate", None)
                    loss_keys = list(self.network.get_loss_keys())
                    loss_val = None
                    for key in loss_keys:
                        if key in batch:
                            loss_val = batch[key]
                            break
                    if isinstance(loss_val, mx.array):
                        loss_val = float(loss_val.item())
                    steps_per_sec = 1.0 / max(time.time() - batch_start, 1e-8)
                    print(
                        f"FastMLX-Train: step: {step}; {loss_keys[0] if loss_keys else 'loss'}: {loss_val}; "
                        f"model_lr: {lr}; steps/sec: {steps_per_sec:.2f};"
                    )
            for t in self.traces:
                if hasattr(t, "on_epoch_end"):
                    t.on_epoch_end(state)
            epoch_time = time.time() - epoch_start
            print(f"FastMLX-Train: step: {step}; epoch: {epoch+1}; epoch_time: {epoch_time:.2f} sec;")
        return state

    def test(self) -> MutableMapping[str, object]:
        """Evaluate the network."""

        state: MutableMapping[str, object] = {"mode": "eval", "metrics": {}}
        for t in self.traces:
            if hasattr(t, "on_epoch_begin"):
                t.on_epoch_begin(state)
        for batch in self.pipeline.get_loader("eval"):
            self.network.run(batch, state)
            for t in self.traces:
                if hasattr(t, "on_batch_end"):
                    t.on_batch_end(batch, state)
        for t in self.traces:
            if hasattr(t, "on_epoch_end"):
                t.on_epoch_end(state)
        print(f"Test metrics: {state['metrics']}")
        return state
