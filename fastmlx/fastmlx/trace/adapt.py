class LRScheduler:
    def __init__(self, model, lr_fn):
        self.model = model
        self.lr_fn = lr_fn
        self.step = 0

    def on_batch_end(self, batch, state):
        self.step += 1
        lr = self.lr_fn(self.step)
        self.model.optimizer.learning_rate = lr
