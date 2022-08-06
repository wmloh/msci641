import torch


def tanh_loss(y_pred, y_true):
    return (-0.5 * ((1 - y_true) * torch.log(1 - y_pred) + (1 + y_true) * torch.log(1 + y_pred)) + 0.6931472).mean()
