import torch


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))