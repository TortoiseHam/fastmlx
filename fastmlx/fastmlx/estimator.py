"""Training and evaluation orchestration."""

from __future__ import annotations

from typing import Iterable, List, MutableMapping, Optional

from .pipeline import Pipeline
from .network import Network


class Estimator:
    """Run a :class:`~fastmlx.pipeline.Pipeline` using a :class:`~fastmlx.network.Network`."""

    def __init__(self, pipeline: Pipeline, network: Network, epochs: int,
                 traces: Optional[Iterable[object]] | None = None) -> None:
        self.pipeline: Pipeline = pipeline
        self.network: Network = network
        self.epochs: int = epochs
        self.traces: List[object] = list(traces or [])

    def fit(self) -> MutableMapping[str, object]:
        """Train the network."""

        state: MutableMapping[str, object] = {}
        for epoch in range(self.epochs):
            state = {"mode": "train", "epoch": epoch, "metrics": {}}
            for t in self.traces:
                if hasattr(t, "on_epoch_begin"):
                    t.on_epoch_begin(state)
            for batch in self.pipeline.get_loader("train"):
                state["batch"] = batch
                self.network.run(batch, state)
                for t in self.traces:
                    if hasattr(t, "on_batch_end"):
                        t.on_batch_end(batch, state)
            for t in self.traces:
                if hasattr(t, "on_epoch_end"):
                    t.on_epoch_end(state)
            print(f"Epoch {epoch+1}: {state['metrics']}")
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
