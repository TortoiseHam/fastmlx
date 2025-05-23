import numpy as np
from ..op import Op

class ExpandDims(Op):
    def __init__(self, inputs, outputs, axis=0):
        super().__init__(inputs, outputs)
        self.axis = axis

    def forward(self, data, state):
        return np.expand_dims(data, axis=self.axis)

class Minmax(Op):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs)

    def forward(self, data, state):
        return data.astype(np.float32) / 255.0
