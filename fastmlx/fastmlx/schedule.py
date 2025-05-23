import math

def cosine_decay(step, cycle_length, init_lr):
    t = (step % cycle_length) / cycle_length
    return init_lr * 0.5 * (1 + math.cos(math.pi * t))
