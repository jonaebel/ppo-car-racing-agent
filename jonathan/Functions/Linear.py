import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Weight matrix and bias — both learnable
        weight = torch.empty(in_features, out_features)
        nn.init.kaiming_uniform_(weight, nonlinearity="relu")
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Just matrix multiplication + bias
        return (x @ self.weight +
                self.bias)