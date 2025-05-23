class Accuracy:
    def __init__(self, true_key="y", pred_key="y_pred"):
        self.true_key = true_key
        self.pred_key = pred_key
        self.correct = 0
        self.total = 0

    def on_epoch_begin(self, state):
        self.correct = 0
        self.total = 0

    def on_batch_end(self, batch, state):
        import numpy as np
        y = batch[self.true_key]
        y_pred = batch[self.pred_key]
        pred = y_pred.argmax(axis=-1)
        self.correct += np.sum(pred == y)
        self.total += len(y)

    def on_epoch_end(self, state):
        state['metrics']["accuracy"] = self.correct / max(1, self.total)
