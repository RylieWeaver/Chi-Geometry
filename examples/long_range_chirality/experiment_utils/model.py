# General
from typing import Dict, Optional

# Torch
import torch
import torch.nn.utils.rnn as rnn_utils

# Equivariant Neural Networks
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from equiformer_pytorch import Equiformer

# Chi-Geometry
from examples.model.e3nn_model import (
    Convolution,
    Compose,
    radius_graph,
    smooth_cutoff,
    tp_path_exists,
)


############################################################################################################
# This is a model taken from the equiformer library by lucidrainsand modified to work with the Chi-Geometry dataset.
# The original model can be found here: https://github.com/lucidrains/equiformer-pytorch
# Accessed: Jan 14th, 2025
############################################################################################################
class WrappedEquiformer(torch.nn.Module):
    """
    A simple Equiformer-based model for classification.
    Uses three degrees (l=0,1,2) with specified hidden dimensions,
    then maps the degree-0 output to 'num_classes'.

    There is reshaping and masking logic because transformers expect
    a batch dimension with constant graph size.
    """

    def __init__(self, input_dim, hidden_dim, layers, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.num_classes = num_classes

        self.input_embedding = torch.nn.Linear(
            118, self.hidden_dim
        )  # Embed one-hot atom type
        # Equiformer setup:
        #  - degree-0 dimension = hidden_dim
        #  - degree-1 dimension = hidden_dim//2
        #  - degree-2 dimension = hidden_dim//8
        self.equiformer = Equiformer(
            dim=(self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 8),
            num_degrees=3,  # l=0,1,2
            depth=self.layers,
            num_linear_attn_heads=1,
            reduce_dim_out=False,
            attend_self=True,
            heads=(4, 1, 1),
            dim_head=(self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 8),
        )
        self.output_embedding = torch.nn.Linear(
            self.hidden_dim, self.num_classes
        )  # Embed down for classification

    def reshape_batch(
        self, feats: torch.Tensor, coords: torch.Tensor, batch: torch.Tensor
    ):
        """
        feats:  [total_nodes, feature_dim]  (concatenated from multiple graphs)
        coords: [total_nodes, 3]           (xyz positions for each node)
        batch:  [total_nodes]              (which graph each node belongs to, in range [0..B-1])

        Returns:
        feats_3d:  [B, N_max, feature_dim]
        coords_3d: [B, N_max, 3]
        mask:      [B, N_max] (True where node is real, False where padded)
        """
        device = feats.device

        # 1) Figure out how many graphs we have
        num_graphs = batch.max().item() + 1

        # 2) Split feats and coords by graph
        feats_per_graph = []
        coords_per_graph = []
        lengths = []
        for g in range(num_graphs):
            node_indices = (batch == g).nonzero(as_tuple=True)[0]
            feats_per_graph.append(feats[node_indices])  # shape [n_g, feature_dim]
            coords_per_graph.append(coords[node_indices])  # shape [n_g, 3]
            lengths.append(len(node_indices))  # store how many nodes in this graph

        # 3) Pad each list to get a uniform node dimension = N_max
        #    feats_3d => [B, N_max, feature_dim]
        #    coords_3d => [B, N_max, 3]
        feats_3d = rnn_utils.pad_sequence(
            feats_per_graph, batch_first=True, padding_value=0.0
        )
        coords_3d = rnn_utils.pad_sequence(
            coords_per_graph, batch_first=True, padding_value=0.0
        )

        # 4) Build a boolean mask of shape [B, N_max], marking real vs. padded
        N_max = feats_3d.size(1)
        mask = torch.zeros((num_graphs, N_max), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        return feats_3d.to(device), coords_3d.to(device), mask

    def forward(self, data):
        # 0) Unpack data
        feats, coords, mask = self.reshape_batch(data.x, data.pos, data.batch)
        # 1) Embed down to hidden_dim
        feats = self.input_embedding(feats)  # [B, N, input_dim]
        # 2) Equiformer
        out = self.equiformer(
            feats,
            coords,
            mask=mask,  # shape [B, N]
        )
        # 3) Use degree-0 outputs for classification
        node_embs = out.type0  # [B, N, hidden_dim]
        logits = self.output_embedding(node_embs)  # [B, N, num_classes]
        # 4) Reshape and mask to expected node outputs
        logits = logits.view(-1, self.num_classes)  # [B*N, num_classes]
        logits = logits[mask.view(-1)].unsqueeze(
            0
        )  # [1, total_valid_nodes, num_classes]

        return logits


############################################################################################################
# This is a model taken from the e3nn library and modified to work with the Chi-Geometry dataset.
# The original model can be found here: https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/gate_points_2101.py
# Accessed: November 6th, 2024
############################################################################################################

"""model with self-interactions and gates

Exact equivariance to :math:`E(3)`

version of january 2021
"""


class CustomNetwork(torch.nn.Module):
    r"""equivariant neural network

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features

    irreps_hidden : `e3nn.o3.Irreps`
        representation of the hidden features

    irreps_out : `e3nn.o3.Irreps`
        representation of the output features

    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials

    layers : int
        number of gates (non linearities)

    max_radius : float
        maximum radius for the convolution

    number_of_basis : int
        number of basis on which the edge length are projected

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes at a distance ``max_radius``

    num_nodes : float
        typical number of nodes in a graph
    """

    def __init__(
        self,
        irreps_in: Optional[o3.Irreps],
        irreps_hidden: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: Optional[o3.Irreps],
        layers: int,
        max_radius: float,
        number_of_basis: int,
        radial_layers: int,
        radial_neurons: int,
        num_neighbors: float,
        num_nodes: float,
        reduce_output: bool = True,
        num_classes: int = 3,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = (
            o3.Irreps(irreps_node_attr)
            if irreps_node_attr is not None
            else o3.Irreps("0e")
        )
        self.irreps_edge_attr_created = o3.Irreps(irreps_edge_attr)
        self.irreps_edge_attr_passed = o3.Irreps(
            "3x0e"
        )  # One scalar indicating the hop distance
        self.irreps_edge_attr = (
            o3.Irreps(self.irreps_edge_attr_passed) + self.irreps_edge_attr_created
        ).simplify()

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.one_hot_embedding = torch.nn.Linear(118, irreps_in.count((0, 1)))

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
            irreps = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
        )

        self.OutModule = torch.nn.ModuleList()
        for _ in range(self.num_heads):
            self.out_embed = torch.nn.Linear(
                self.irreps_out.count((0, 1)) + self.irreps_out.count((0, -1)),
                self.num_classes,
            )
            self.OutModule.append(self.out_embed)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """evaluate the network

        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        edge_index = data.edge_index
        if "edge_index" not in data or edge_index is None:
            edge_index = radius_graph(data["pos"], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge_attr_created, edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh
        full_edge_attr = torch.cat([data.edge_attr, edge_attr], dim=-1)

        if self.input_has_node_in and "atomic_numbers_one_hot" in data:
            assert self.irreps_in is not None
            x = data["atomic_numbers_one_hot"]
            x = self.one_hot_embedding(x)
        elif self.input_has_node_in and "x" in data:
            assert self.irreps_in is not None
            x = data["x"]
        else:
            assert self.irreps_in is None
            x = data["pos"].new_ones((data["pos"].shape[0], 1))

        if self.input_has_node_attr and "z" in data:
            z = data["z"]
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data["pos"].new_ones((data["pos"].shape[0], 1))

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, full_edge_attr, edge_length_embedded)

        outputs = []
        for out_embed in self.OutModule:
            out = out_embed(x)
            outputs.append(out)
        return torch.stack(outputs)

        # if self.reduce_output:
        #     return scatter(x, batch, dim_size=int(batch.max()) + 1).div(self.num_nodes**0.5)
        # else:
        #     return x
