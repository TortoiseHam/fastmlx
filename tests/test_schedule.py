from fastmlx.schedule import cosine_decay


def test_cosine_decay():
    lr = cosine_decay(step=0, cycle_length=10, init_lr=1.0)
    assert abs(lr - 1.0) < 1e-6
    lr_mid = cosine_decay(step=5, cycle_length=10, init_lr=1.0)
    assert lr_mid < 1.0
