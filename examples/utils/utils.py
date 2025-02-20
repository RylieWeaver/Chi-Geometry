# General
import json
import networkx as nx

# Torch
import torch
from torch_scatter import scatter
from torch_geometric.utils import degree


def load_model_json(file_path):
    # Default arguments
    default_args = {
        # Learning parameters
        "epochs": 75,
        "batch_size": 128,
        "lr": 0.001,
        "use_cuda": False,
        # Model parameters
        "input_dim": "40",
        "hidden_dim": "20",
        "irreps_node_attr": "1x0e",
        "irreps_edge_attr": "1x0e + 1x1o",
        "layers": 4,
        "max_radius": 2.0,
        "number_of_basis": 5,
        "radial_layers": 2,
        "radial_neurons": 50,
        "reduce_output": False,
        "output_dim": 3,
        "class_weights": [1.0, 10.0, 10.0],
    }

    # Read configuration from JSON file
    try:
        with open(file_path, "r") as f:
            config_args = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found at {file_path}. Using default arguments.")
        config_args = {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}. Using default arguments.")
        config_args = {}

    # Merge default arguments with those from the config file
    args = {**default_args, **config_args}

    # Make irreps from dims
    args["irreps_in"] = f"{args['input_dim']}x0e"
    args["irreps_hidden"] = create_irreps_string(int(args["hidden_dim"]), 1)
    args["irreps_out"] = create_irreps_string(int(args["hidden_dim"]), 0)

    return args


def process_batch(data):
    # Center the positions (necessary for E(3) equivariance models)
    mean_pos = scatter(data.pos, data.batch, dim=0, reduce="mean")
    data.pos = data.pos - mean_pos[data.batch]

    return data


def create_irreps_string(node_dim: int, max_ell: int):
    """Create a string representation of the irreps for the E(3) layer"""
    irreps = []
    for ell in range(max_ell + 1):
        irreps.append(f"{node_dim}x{ell}e")
        irreps.append(f"{node_dim}x{ell}o")
    return " + ".join(irreps)


def make_global_connections(dataset):
    for data in dataset:
        num_nodes = data.num_nodes
        edge_pairs = []

        # Get global connections
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_pairs.append([i, j])

        # Overwrite
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        data.edge_index = edge_index

    return dataset


def global_connect_feat_eng(dataset, undirected=True):
    """
    Overwrites each `data` in the dataset with a fully connected (global) edge_index.
    Also creates data.edge_attr for each edge that includes:
      - BFS distance in the original graph
      - is_original_edge (0.0 or 1.0)

    Steps:
      1) Read the *original* edges (data.edge_index).
      2) Build an undirected NetworkX graph G from them.
      3) Use all_pairs_shortest_path_length to get BFS distances for every pair.
      4) Create a new edge_index containing *all* pairs (i, j) where i != j.
      5) For each pair (i, j), record:
          - BFS distance in G (or a default large number if unreachable).
          - 1.0 if (i, j) was in the original, else 0.0.
      6) Store them in data.edge_attr with shape [E, 2].

    If your graph was undirected originally, set `undirected=True`, which ensures
    that an edge (i, j) in the original graph also implies (j, i) is "original."
    """

    for data in dataset:
        # ----------------------------
        # (A) Original edges
        # ----------------------------
        orig_edges = set()
        i_list, j_list = data.edge_index
        i_list, j_list = i_list.tolist(), j_list.tolist()
        for u, v in zip(i_list, j_list):
            orig_edges.add((u, v))
            # If graph is undirected, also store the reverse
            if undirected:
                orig_edges.add((v, u))

        # ----------------------------
        # (B) Build Nx Graph for BFS
        # ----------------------------
        G = nx.Graph() if undirected else nx.DiGraph()
        num_nodes = data.num_nodes
        G.add_nodes_from(range(num_nodes))
        for u, v in zip(i_list, j_list):
            G.add_edge(u, v)

        # All-pairs BFS shortest path lengths
        all_pairs = dict(nx.all_pairs_shortest_path_length(G))

        # ----------------------------
        # (C) Create new global edges
        # ----------------------------
        all_edges = []
        edge_attrs = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                all_edges.append([i, j])

                # BFS distance in the original graph.
                bfs_dist = float(all_pairs[i].get(j, num_nodes + 1))

                # is_original is your one-hot vector [1.0, 0.0] or [0.0, 1.0]
                is_original = [1.0, 0.0] if (i, j) in orig_edges else [0.0, 1.0]

                edge_attrs.append([bfs_dist] + is_original)

        # Convert to tensors
        edge_index = (
            torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        )  # shape [2, E]
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)  # shape [E, 2]

        # Overwrite data.edge_index and data.edge_attr
        data.edge_index = edge_index
        data.edge_attr = edge_attr

    return dataset


def get_avg_degree(dataset):
    total_degree = 0
    total_nodes = 0
    for data in dataset:
        deg = degree(data.edge_index[0], data.num_nodes)  # per-node degrees
        total_degree += deg.sum().item()
        total_nodes += data.num_nodes

    return total_degree / total_nodes if total_nodes > 0 else 1.0


def get_avg_nodes(dataset):
    total_nodes = 0
    for data in dataset:
        total_nodes += data.num_nodes
    return total_nodes / len(dataset) if len(dataset) > 0 else 1.0


class WrappedModel(torch.nn.Module):
    def __init__(self, base, input_dim, output_dim, avg_nodes=5):
        super().__init__()
        self.base = base
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = torch.nn.Linear(5 * input_dim, output_dim)
        self.avg_nodes = avg_nodes

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
        std_val = (mean_val_sq - x_mean ** 2).clamp_min(1e-9).sqrt()
        x = torch.cat(
            [x_sum.div(self.avg_nodes ** 0.5), x_mean, x_min, x_max, std_val], dim=1
        )

        return self.embedding(x)


# Updates
## Take num neighbors and num nodes out of e3nn model
## Clamp distance in e3nn model
## Take out num heads in framework
## Consolidate training and models to work for classification and regression
## Take out reduce output in e3nn model
## Change class tags from float to long
## Update training so no multihead
## References to num_classes in the examples
## Renamed from model to utils
## Will need to fully rework imports and calls in simple, crossed, etc... examples
## Changing the training function to take train/val/test instead of the whole dataset
