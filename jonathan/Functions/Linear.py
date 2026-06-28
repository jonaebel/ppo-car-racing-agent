import torch


class Linear:
    def __init__(self, in_features, out_features):
        # Weight matrix and bias — both learnable
        self.weight = torch.nn.Parameter(
            torch.randn(in_features, out_features) * 0.01
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(out_features)
        )

    def forward(self, x):
        # Just matrix multiplication + bias
        return x @ self.weight + self.bias