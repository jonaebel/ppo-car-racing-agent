# Visual Encoder layer -> Convolutional Network
# Gets 96 x 96 RGB pixel grid as input and converts them into one 512 dimensional Vector which
# gets presented to the decion making instance (ppo_agent.py)
# https://cs231n.github.io/convolutional-networks/

# K —> how many filters (= depth of output)
# F —> filter size (3x3, 5x5 etc.)
# S —> stride, how many pixels the filter jumps each step
# P —> zero padding, extra zeros added around the border

#output size = (W - F + 2P) / S + 1


# INPUT (4, 96, 96)

# → CONV → RELU        detects edges
# → CONV → RELU        detects road shape
# → CONV → RELU        detects track ahead
# → CONV → RELU        derects what else is needed

# → FLATTEN
# → FC → RELU          compress to 512
# → output to ppo_agent.py

import torch
import torch.nn.functional as F

from jonathan.Functions import ReLu
from jonathan.Functions.Linear import Linear
from jonathan.layers import Conv2d


class CNNFeatureExtractor:
    def __init__(self):
        self.conv1 = Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = Linear(4096, 512)

    def forward(self, x):
        x = ReLu(self.conv1.forward(x))
        x = ReLu(self.conv2.forward(x))
        x = ReLu(self.conv3.forward(x))
        x = x.flatten(start_dim=1)      # (batch, 4096)
        x = ReLu(self.linear.forward(x)) # (batch, 512)
        return x

    def parameters(self):
        # Collect all learnable parameters for the optimizer    
        return (
            list(self.conv1.weight) + [self.conv1.bias] +
            list(self.conv2.weight) + [self.conv2.bias] +
            list(self.conv3.weight) + [self.conv3.bias] +
            [self.linear.weight, self.linear.bias]
        )