# General
import math
from math import sqrt
from typing import Callable

# PyTorch
import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.typing import Tensor


class EmbeddingBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim: int, act: Callable):
        super().__init__()
        self.emb = nn.Linear(input_dim, hidden_dim)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.emb(x))


class Convolution(nn.Module):
    def __init__(self, hidden_dim: int, act: Callable, avg_degree: int = 1):
        super().__init__()
        self.interact = nn.Linear(2*hidden_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.act = act
        self.avg_degree = avg_degree

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        m_ij = self.act(self.interact(torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)))
        m_i = scatter(m_ij, edge_index[0], dim=0, dim_size=x.size(0), reduce='sum') / sqrt(self.avg_degree)
        x = x + self.act(self.update(x + m_i))
        return x
    

class OutputBlock(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(x)


class VanillaMPNN(nn.Module):
    """
    A basic message-passing neural network (MPNN) model for atomic structural property prediction.
    """

    def __init__(
        self,
        hidden_dim: int = 20,
        layers: int = 4,
        avg_degree: int = 1,
        output_dim: int = 3,
    ):
        super().__init__()
        # Hyperparameters
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.avg_degree = avg_degree
        self.output_dim = output_dim
        # Defaults that we just set for simplicity
        self.act = nn.SiLU()

        # 1) Linearly embed the one-hot atomic numbers
        self.input_embed = EmbeddingBlock(118+3, hidden_dim, self.act)  # 118 is the number of elements in the periodic table, +3 for 3D coordinates

        # 2) Convolutional blocks
        self.interaction_blocks = nn.ModuleList()
        for _ in range(layers):
            self.interaction_blocks.append(
                Convolution(hidden_dim, self.act, self.avg_degree)
            )
        
        # 3) Output block
        self.output_block = OutputBlock(hidden_dim, output_dim)


    def forward(
        self,
        data,
    ) -> torch.Tensor:
        """
        data.atomic_numbers_one_hot:     [num_nodes, one_hot_dim]   (one-hot atom features)
        data.pos:   [num_nodes, 3]             (3D coordinates)
        """
        # 1) Embeddings the one-hot atomic numbers
        x = self.input_embed(
            torch.cat([data.atomic_numbers_one_hot, data.pos], dim=-1),
        )  # shape: [num_nodes, 118+3] --> [num_nodes, hidden_dim]
        sc = x

        # 2) Convolutional blocks
        for conv in self.interaction_blocks:
            x = conv(x, data.edge_index)
        x = sc + x  # Residual connection
        
        # 3) Output block
        x = self.output_block(x)  # shape: [num_nodes, hidden_dim] --> [num_nodes, output_dim]

        return x
