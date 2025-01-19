# General
from typing import Dict, Optional

# Torch
import torch
import torch.nn.utils.rnn as rnn_utils
from torch_scatter import scatter

# Chi-Geometry
from chi_geometry.model import Network


class Wrapped_Network(torch.nn.Module):
    def __init__(self, base, input_dim, output_dim):
        super().__init__()
        self.base = base
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = torch.nn.Linear(5 * input_dim, output_dim)

    def forward(self, data):
        # Base network
        x = self.base(data).squeeze()  # General squeeze for redundant dimensions

        # sum, mean, min, max
        x_sum = scatter(x, data.batch, dim=0, reduce="sum")
        x_mean = scatter(x, data.batch, dim=0, reduce="mean")
        x_min = scatter(x, data.batch, dim=0, reduce="min")
        x_max = scatter(x, data.batch, dim=0, reduce="max")

        # manual std
        mean_val_sq = scatter(x ** 2, data.batch, dim=0, reduce="mean")
        std_val = (mean_val_sq - x_mean ** 2).clamp_min(1e-5).sqrt()
        x = torch.cat([x_sum, x_mean, x_min, x_max, std_val], dim=1)

        return self.embedding(x)
