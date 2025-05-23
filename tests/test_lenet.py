import mlx.core as mx
from fastmlx.architecture import LeNet


def test_lenet_forward():
    model = LeNet()
    mx.eval(model.parameters())
    x = mx.random.uniform(shape=(2, 28, 28, 1))
    y = model(x)
    assert y.shape == (2, 10)

