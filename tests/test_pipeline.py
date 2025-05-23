import numpy as np
import fastmlx as fe
from fastmlx.dataset import NumpyDataset
from fastmlx.op.numpyop import ExpandDims, Minmax


def test_pipeline_ops():
    data = NumpyDataset({'x': np.zeros((4, 28, 28), dtype=np.uint8)})
    pipe = fe.Pipeline(train_data=data, batch_size=2,
                       ops=[ExpandDims('x', 'x', axis=-1), Minmax('x', 'x')])
    loader = pipe.get_loader('train')
    batch = next(iter(loader))
    assert batch['x'].shape == (2, 28, 28, 1)
    assert np.allclose(batch['x'], 0.0)
