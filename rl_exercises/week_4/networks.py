from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A simple MLP mapping state → Q‐values for each action.

    Architecture:
      Input → Linear(obs_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→n_actions)
    """

    def __init__(
        self, obs_dim: int, n_actions: int, hidden_dims: List[int] = [64, 64]
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_dims : List[int]
            List of sizes for each hidden layer.
        """
        super().__init__()

        layers = OrderedDict()
        current_dim = obs_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers[f"fc{i+1}"] = nn.Linear(current_dim, hidden_dim)
            layers[f"relu{i+1}"] = nn.ReLU()
            current_dim = hidden_dim

        layers["out"] = nn.Linear(current_dim, n_actions)

        self.net = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, obs_dim).

        Returns
        -------
        torch.Tensor
            Q‐values, shape (batch, n_actions).
        """
        return self.net(x)
