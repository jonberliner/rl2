import torch

def normalized_sigmoid(x, start=0., end=1., norm_start=-6, norm_end=6.):
    """will find the value of the _sigmoid(x), where _sigmoid is a
    scaled and shifted function such that _sigmoid(start) = sigmoid(norm_start),
    and _sigmoid(end) = sigmoid(norm_end)"""
    norm_scale = norm_end - norm_start

    scale = end - start
    x0 = (x - start) / scale

    x1 = (x0 * norm_scale) + norm_start
    return 1. / (1. + torch.exp(-x1))


def laggy_logistic_fn(init_lag, sigmoid_lag, end_lag, fps=24.):
    total_time = init_lag + sigmoid_lag + end_lag
    _total_time = total_time.item()
    x = torch.linspace(0, _total_time, int(_total_time * fps))
    y = normalized_sigmoid(x, init_lag, init_lag + sigmoid_lag)
    return x, y

