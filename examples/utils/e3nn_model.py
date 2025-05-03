############################################################################################################
# This is a model taken from the e3nn library and modified to work with the Chi-Geometry dataset.
# The original model can be found here: https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/gate_points_2101.py
# Accessed: November 6th, 2024
############################################################################################################

"""model with self-interactions and gates

Exact equivariance to :math:`E(3)`

version of january 2021
"""
import math
from typing import Dict, Optional

import torch

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, Activation
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode


def radius_graph(pos, r_max, batch) -> torch.Tensor:
    # naive and inefficient version of torch_cluster.radius_graph
    r = torch.cdist(pos, pos)
    index = ((r < r_max) & (r > 0)).nonzero().T
    index = index[:, batch[index[0]] == batch[index[1]]]
    return index


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)


@compile_mode("script")
class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_out : `e3nn.o3.Irreps` or None
        representation of the output node features

    number_of_basis : int
        number of basis on which the edge length are projected

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    avg_degree : float
        typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        number_of_basis,
        radial_layers,
        radial_neurons,
        avg_degree,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.avg_degree = avg_degree

        self.sc = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node_attr, self.irreps_out
        )
        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node_attr, self.irreps_in
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            [number_of_basis] + radial_layers * [radial_neurons] + [tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node_attr, self.irreps_out
        )

    def forward(
        self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded
    ) -> torch.Tensor:
        weight = self.fc(edge_length_embedded)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_features, edge_dst, dim_size=x.shape[0]).div(
            self.avg_degree ** 0.5
        )

        x = self.lin2(x, node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x


def smooth_cutoff(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


############################################################################################################


class Network(torch.nn.Module):
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

    avg_degree : float
        typical number of nodes convolved over
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
        avg_degree: int,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.eps = 1e-9
        self.number_of_basis = number_of_basis
        self.avg_degree = avg_degree
        self.output_dim = output_dim

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = (
            o3.Irreps(irreps_node_attr)
            if irreps_node_attr is not None
            else o3.Irreps("0e")
        )
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

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
                avg_degree,
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
                avg_degree,
            )
        )

        self.out_embed = torch.nn.Linear(
            self.irreps_out.count((0, 1)) + self.irreps_out.count((0, -1)),
            self.output_dim,
        )

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
            self.irreps_edge_attr, edge_vec, True, normalization="component"
        )
        edge_length = torch.clamp(
            edge_vec.norm(dim=1), min=self.eps, max=self.max_radius - self.eps
        )
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis ** 0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

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
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        return self.out_embed(x)


############################################################################################################


"""model with self-interactions and gates

Exact equivariance to :math:`E(3)`

version of january 2021
"""


class CustomNetwork(torch.nn.Module):
    # Customized to include invariant edge attributes
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

    avg_degree : float
        typical number of nodes convolved over
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
        avg_degree: int,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.eps = 1e-9
        self.number_of_basis = number_of_basis
        self.output_dim = output_dim

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
                avg_degree,
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
                avg_degree,
            )
        )

        self.out_embed = torch.nn.Linear(
            self.irreps_out.count((0, 1)) + self.irreps_out.count((0, -1)),
            self.output_dim,
        )

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
        edge_length = torch.clamp(
            edge_vec.norm(dim=1), min=self.eps, max=self.max_radius - self.eps
        )
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis ** 0.5)
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

        return self.out_embed(x)
    

############################################################################################################

class VirtualCompose(torch.nn.Module):
    def __init__(self, first, second, third) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.third = third
        self.virtual_irreps_in = self.first.virtual_irreps_in
        self.node_irreps_in = self.first.node_irreps_in
        self.irreps_out = [self.second.irreps_out, self.third.irreps_out]

    def forward(self, *input):
        x_virtual, x_node = self.first(*input)
        return self.second(x_virtual), self.third(x_node)


@compile_mode("script")
class VirtualNodeConvolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    virtual_irreps_in : `e3nn.o3.Irreps`
        representation of the input virtual node features

    node_irreps_in : `e3nn.o3.Irreps`
        representation of the input node features

    virtual_irreps_out : `e3nn.o3.Irreps`
        representation of the output virtual node features

    node_irreps_out : `e3nn.o3.Irreps`
        representation of the output node features

    avg_degree : float
        typical number of nodes convolved over

    avg_nodes : float
        typical number of nodes in the graph
    """

    def __init__(
        self,
        virtual_irreps_in,
        node_irreps_in,
        virtual_irreps_out,
        node_irreps_out,
        node_pos_irreps,
        avg_nodes,
    ) -> None:
        super().__init__()
        self.virtual_irreps_in = o3.Irreps(virtual_irreps_in)
        self.node_irreps_in = o3.Irreps(node_irreps_in)
        self.virtual_irreps_out = o3.Irreps(virtual_irreps_out)
        self.node_irreps_out = o3.Irreps(node_irreps_out)
        self.node_pos_irreps = o3.Irreps(node_pos_irreps)
        self.avg_nodes = avg_nodes

        # Projected skip connections
        self.virtual_sc = o3.Linear(self.virtual_irreps_in, self.virtual_irreps_out)
        self.node_sc = o3.Linear(self.node_irreps_in, self.node_irreps_out)

        # Messages
        self.v2n_tp = FullyConnectedTensorProduct(
            self.node_irreps_in,
            self.node_pos_irreps,
            self.virtual_irreps_out,
        )
        self.n2v_lin = o3.Linear(self.virtual_irreps_out, self.node_irreps_out)

        # Activation layers
        ## Virtual
        virtual_acts = []
        for (mul, ir) in self.virtual_irreps_out:
            if ir.l == 0:
                # It's a scalar block
                if ir.p == 1:
                    # even scalar (0e)
                    virtual_acts.append(torch.nn.SiLU())
                elif ir.p == -1:
                    # odd scalar (0o) – must use an odd function (like Tanh)
                    virtual_acts.append(torch.nn.Tanh())
                else:
                    # p=0 is invalid for scalars in e3nn
                    raise ValueError(f"Unexpected parity {ir.p} for scalar {ir}")
            else:
                # It's non-scalar (l>0): no direct activation
                virtual_acts.append(None)
        self.act_v = Activation(self.virtual_irreps_out, virtual_acts)
        ## Node
        node_acts = []
        for (mul, ir) in self.node_irreps_out:
            if ir.l == 0:
                # It's a scalar block
                if ir.p == 1:
                    # even scalar (0e)
                    node_acts.append(torch.nn.SiLU())  # or ReLU, etc.
                elif ir.p == -1:
                    # odd scalar (0o)
                    node_acts.append(torch.nn.Tanh())  # Tanh is an odd function
                else:
                    raise ValueError(f"Unexpected parity {ir.p} for scalar {ir}")
            else:
                # It's non-scalar (l>0), no direct activation
                node_acts.append(None)
        self.act_n = Activation(self.node_irreps_out, node_acts)

    def forward(
        self, x_virtual, x_node, node_pos_sh, batch
    ) -> torch.Tensor:
        # Skip connections
        s_virtual = self.virtual_sc(x_virtual)
        s_node = self.node_sc(x_node)

        # Update virtual with node features
        ## Compute message
        m_n2v = self.v2n_tp(x_node, node_pos_sh)
        m_n2v = scatter(m_n2v, batch, dim_size=x_virtual.shape[0]).div(self.avg_nodes ** 0.5)
        m_n2v = self.act_v(m_n2v)
        ## Skip connection
        x_virtual = (s_virtual + m_n2v) / (2 ** 0.5)

        # Update node with virtual features
        ## Compute messages
        m_v2n = self.n2v_lin(x_virtual[batch])
        m_v2n = self.act_n(m_v2n)
        ## Skip connection
        x_node = (s_node + m_v2n) / (2 ** 0.5)
        
        return x_virtual, x_node


class VirtualNodeNetwork(torch.nn.Module):
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

    avg_degree : float
        typical number of nodes convolved over
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
        avg_degree: int,
        avg_nodes: int,
        max_pos_norm: float,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.eps = 1e-9
        self.number_of_basis = number_of_basis
        self.avg_degree = avg_degree
        self.avg_nodes = avg_nodes
        self.max_pos_norm = max_pos_norm
        self.output_dim = output_dim

        self.irreps_in = irreps_in
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = (
            o3.Irreps(irreps_node_attr)
            if irreps_node_attr is not None
            else o3.Irreps("0e")
        )
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_pos = o3.Irreps(irreps_edge_attr)

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None

        node_irreps = (self.irreps_in + self.irreps_node_pos)
        virtual_irreps = node_irreps

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.one_hot_embedding = torch.nn.Linear(118, self.irreps_in.count((0, 1)))

        self.layers = torch.nn.ModuleList()
        self.global_layers = torch.nn.ModuleList()

        for _ in range(layers):
            # Global convolution
            ## Virtual Gate
            virtual_irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in node_irreps
                    if ir.l == 0 and tp_path_exists(virtual_irreps, node_irreps, ir)
                ]
            )
            virtual_irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in node_irreps
                    if ir.l > 0 and tp_path_exists(virtual_irreps, node_irreps, ir)
                ]
            )
            virtual_ir = "0e" if tp_path_exists(virtual_irreps, node_irreps, "0e") else "0o"
            virtual_irreps_gates = o3.Irreps([(mul, virtual_ir) for mul, _ in virtual_irreps_gated])
            virtual_conv_gate = Gate(
                virtual_irreps_scalars,
                [act[ir.p] for _, ir in virtual_irreps_scalars],  # scalar
                virtual_irreps_gates,
                [act_gates[ir.p] for _, ir in virtual_irreps_gates],  # gates (scalars)
                virtual_irreps_gated,  # gated tensors
            )
            ## Node Gate
            node_irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in node_irreps
                    if ir.l == 0 and tp_path_exists(node_irreps, virtual_irreps, ir)
                ]
            )
            node_irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in node_irreps
                    if ir.l > 0 and tp_path_exists(node_irreps, virtual_irreps, ir)
                ]
            )
            node_ir = "0e" if tp_path_exists(node_irreps, virtual_irreps, "0e") else "0o"
            node_irreps_gates = o3.Irreps([(mul, node_ir) for mul, _ in node_irreps_gated])
            node_conv_gate = Gate(
                node_irreps_scalars,
                [act[ir.p] for _, ir in node_irreps_scalars],  # scalar
                node_irreps_gates,
                [act_gates[ir.p] for _, ir in node_irreps_gates],  # gates (scalars)
                node_irreps_gated,  # gated tensors
            )
            global_conv = VirtualNodeConvolution(
                virtual_irreps,
                node_irreps,
                virtual_conv_gate.irreps_in,
                node_conv_gate.irreps_in,
                self.irreps_edge_attr,
                avg_nodes,
            )
            virtual_irreps = virtual_conv_gate.irreps_out
            node_irreps = node_conv_gate.irreps_out
            self.global_layers.append(VirtualCompose(global_conv, virtual_conv_gate, node_conv_gate))

            # Standard Convolution
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and tp_path_exists(node_irreps, self.irreps_edge_attr, ir)
                ]
            )
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l > 0 and tp_path_exists(node_irreps, self.irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(node_irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            conv_gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                node_irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                conv_gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                avg_degree,
            )
            node_irreps = conv_gate.irreps_out
            self.layers.append(Compose(conv, conv_gate))

        # Final convolutions
        self.global_layers.append(
            VirtualNodeConvolution(
                virtual_irreps,
                node_irreps,
                virtual_irreps,
                node_irreps,
                self.irreps_edge_attr,
                avg_nodes,
            )
        )
        self.layers.append(
            Convolution(
                node_irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                avg_degree,
            )
        )

        self.out_embed = torch.nn.Linear(
            self.irreps_out.count((0, 1)) + self.irreps_out.count((0, -1)),
            self.output_dim,
        )

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
            self.irreps_edge_attr, edge_vec, True, normalization="component"
        )
        edge_length = torch.clamp(
            edge_vec.norm(dim=1), min=self.eps, max=self.max_radius - self.eps
        )
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis ** 0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        node_vec = data["pos"]
        node_sh = o3.spherical_harmonics(
            self.irreps_edge_attr, node_vec, True, normalization="component"
        )
        node_length = torch.clamp(
            node_vec.norm(dim=1), min=self.eps, max=self.max_pos_norm - self.eps
        )
        node_pos_sh = smooth_cutoff(node_length / self.max_pos_norm)[:, None] * node_sh

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

        # Add node pos irreps to node irreps
        x = torch.cat([x, node_pos_sh], dim=-1)

        # Initialize virtual node as zeros
        x_virtual = torch.zeros((data.batch.max().item()+1, x.shape[1]), device=x.device, dtype=x.dtype)

        # Perform convolutions
        for lay, global_lay in zip(self.layers, self.global_layers):
            x_virtual, x = global_lay(x_virtual, x, node_pos_sh, data.batch)
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        return self.out_embed(x)



# WORKS NOTES
## Setup
### 4 layers, dims [40, 20, 3]
### All samples
### pos_irreps not catted to node_irreps
## Network
### all TPs... having the sc TP seemed necessary
## VirtualNodeConvolution
### no linear message (only TP message)
### m_n2v = self.v2n_tp(x_node, node_pos_sh)
### m_v2n = self.n2v_lin(x_virtual[batch])
### no tensor product normalization
### yes activations with Tanh
## Observations
### Signs clearly shown on epoch 3, maybe even on 2
