import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(weight, nonlinearity="relu")
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride)

    # forward calculations made by PyTorch cause its more efficent
