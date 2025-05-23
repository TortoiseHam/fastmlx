class Trace:
    """Minimal Trace base class following FastEstimator semantics."""

    def on_start(self, state):
        pass

    def on_epoch_begin(self, state):
        pass

    def on_batch_begin(self, batch, state):
        pass

    def on_batch_end(self, batch, state):
        pass

    def on_epoch_end(self, state):
        pass

    def on_finish(self, state):
        pass
