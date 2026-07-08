"""PyTorch Q-network for DQN."""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class QNetwork(nn.Module if nn is not None else object):
    """Simple MLP mapping state vectors to action values."""
    def __init__(self, state_dim, action_dim, hidden_dims=(128, 128)):
        if nn is None:
            raise ImportError("PyTorch is required for QNetwork")
        super().__init__()
        layers = []
        prev = state_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU()])
            prev = hidden
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
