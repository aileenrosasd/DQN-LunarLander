from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F 

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, int] = (128, 128)):
        # define a fully connected neural network
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass through the network 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
