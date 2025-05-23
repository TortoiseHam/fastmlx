from typing import List
from .pipeline import Pipeline
from .network import Network
from .trace.metric import Accuracy


class Estimator:
    def __init__(self, pipeline: Pipeline, network: Network, epochs: int, traces: List=None):
        self.pipeline = pipeline
        self.network = network
        self.epochs = epochs
        self.traces = traces or []

    def fit(self):
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

    def test(self):
        state = {"mode": "eval", "metrics": {}}
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
