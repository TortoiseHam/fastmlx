class NumpyDataset:
    def __init__(self, data):
        self.data = data
        self.size = len(next(iter(data.values())))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
