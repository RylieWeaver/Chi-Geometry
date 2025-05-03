# General
import math
from math import sqrt
from typing import Callable

# PyTorch
import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.typing import Tensor
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    SphericalBasisLayer,
    triplets,
    InteractionPPBlock,
    OutputPPBlock,
)


###################################################################
# DimeNet++ Implementation based on PyG:
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html#DimeNetPlusPlus
###################################################################
class PairwiseEmbeddingBlock(torch.nn.Module):
    def __init__(self, input_dim, num_radial: int, hidden_dim: int, act: Callable):
        super().__init__()
        self.act = act
        self.emb = nn.Linear(input_dim, hidden_dim)
        self.lin_rbf = nn.Linear(num_radial, hidden_dim)
        self.lin = nn.Linear(3 * hidden_dim, hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class DimeNetPP(nn.Module):
    """
    A lightweight DimeNet++-style model for node-level (or graph-level) tasks,
    using a custom one-hot embedding for atomic numbers.

    Skips the "pretrained QM9" code and uses only the bare essentials:
      - Bessel/Spherical basis
      - InteractionPPBlock
      - OutputPPBlock
      - Basic radius_graph + triplets
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 20,
        layers: int = 4,
        output_dim: int = 3,
        max_radius: float = 5.0,
    ):
        super().__init__()
        # Hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.output_dim = output_dim
        self.max_radius = max_radius
        # Defaults that we just set for simplicity
        self.act = nn.SiLU()
        num_radial = 5
        num_spherical = 3
        int_emb_dim = math.ceil(hidden_dim / 4)
        basis_emb_dim = math.ceil(hidden_dim / 4)
        out_emb_dim = math.ceil(hidden_dim / 4)
        num_before_skip = 1
        num_after_skip = 1
        num_output_layers = 1

        # 1) Linearly embed the one-hot atomic numbers
        self.input_embed = nn.Linear(118, input_dim)

        # 2) DimeNet++ basis layers (Bessel / Spherical)
        self.rbf = BesselBasisLayer(
            num_radial=5,
            cutoff=self.max_radius,
            envelope_exponent=5,
        )
        self.sbf = SphericalBasisLayer(
            num_spherical=3,
            num_radial=5,
            cutoff=self.max_radius,
            envelope_exponent=5,
        )

        # 3) Pairwise embedding block
        self.pairwise_embed = PairwiseEmbeddingBlock(
            input_dim, num_radial, hidden_dim, self.act
        )

        # 4) DimeNet++ Interaction and Output blocks
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionPPBlock(
                    hidden_channels=hidden_dim,
                    int_emb_size=int_emb_dim,
                    basis_emb_size=basis_emb_dim,
                    num_spherical=num_spherical,
                    num_radial=num_radial,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    act=self.act,
                )
                for _ in range(layers)
            ]
        )

        self.output_blocks = nn.ModuleList(
            [
                OutputPPBlock(
                    num_radial=num_radial,
                    hidden_channels=hidden_dim,
                    out_emb_channels=out_emb_dim,
                    out_channels=output_dim,
                    num_layers=num_output_layers,
                    act=self.act,
                    output_initializer="zeros",
                )
                for _ in range(layers + 1)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.rbf.reset_parameters()
        self.pairwise_embed.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def forward(
        self,
        data,
    ) -> torch.Tensor:
        """
        data.atomic_numbers_one_hot:     [num_nodes, one_hot_dim]   (one-hot atom features)
        data.pos:   [num_nodes, 3]             (3D coordinates)
        data.batch: [num_nodes]                (which graph each node belongs to)
        """
        # 0) Feature Processing
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            data.edge_index, num_nodes=data.x.size(0)
        )
        # Pairwise distances
        dist = (data.pos[i] - data.pos[j]).norm(dim=-1)
        # Triplet angles
        pos_jk = data.pos[idx_j] - data.pos[idx_k]
        pos_ij = data.pos[idx_i] - data.pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)
        # RBF and SBF
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # 1) Embeddings the one-hot atomic numbers
        x = self.input_embed(
            data.atomic_numbers_one_hot
        )  # shape: [num_nodes, input_dim] --> [num_nodes, hidden_dim]

        # 2) Embed Pairwise messages
        x = self.pairwise_embed(x, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=data.pos.size(0))

        # Convolutional blocks
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=data.pos.size(0))

        return P
