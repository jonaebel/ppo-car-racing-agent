import torch
import torch.nn.functional as F

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,):
        # Learnable Filters, Initialisation is random for now
        self.weight = torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * 0.01
        self.bias = torch.zeros(out_channels)

        self.weight = torch.nn.Parameter(self.weight)
        self.bias = torch.nn.Parameter(self.bias)

        self.stride = stride

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride)

    # forward calculations made by PyTorch cause its more efficent
