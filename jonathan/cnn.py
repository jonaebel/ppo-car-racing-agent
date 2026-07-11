# Visual Encoder layer -> Convolutional Network
# CNN Network is based in Atari DQN architecture (https://arxiv.org/pdf/1312.5602) from 2013
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

# → FLATTEN
# → FC → RELU          compress to 512
# → output to ppo_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    def __init__(self, obs_shape=(4, 96, 96)):
        super().__init__()
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Kaiming init for ReLU activations
        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        # Compute flatten size dynamically — stays correct if obs_shape or kernels change
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            n_flat = self._conv_forward(dummy).shape[1]

        self.linear = nn.Linear(n_flat, 512)
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear.bias)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_forward(x)         # (batch, n_flat)
        x = F.relu(self.linear(x))        # (batch, 512)
        return x
