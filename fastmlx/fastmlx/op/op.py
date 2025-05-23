class Op:
    def __init__(self, inputs=None, outputs=None):
        self.inputs = [] if inputs is None else ([inputs] if isinstance(inputs, str) else list(inputs))
        self.outputs = [] if outputs is None else ([outputs] if isinstance(outputs, str) else list(outputs))

    def forward(self, data, state):
        raise NotImplementedError
