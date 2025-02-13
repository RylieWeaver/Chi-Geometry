# General
import os
import datetime
import networkx as nx

# Torch
import torch

# E3NN
from e3nn import o3

# Chi-Geometry
from examples.model import load_model_json
from experiment_utils.utils import create_all_datasets
from experiment_utils.train_val_test import train_val_test_model
from experiment_utils.model import CustomNetwork


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


def main():
    # Setup
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    distances = [8, 7]

    # Create datasets
    # dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    # create_all_datasets(distances, dataset_config_path)

    # Args
    model_config_path = os.path.join(script_dir, "e3nn_global_model_config.json")
    model_args = load_model_json(model_config_path)
    noise = True
    datadir = "datasets"
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    # Train model on each dataset
    for dist in distances:
        # Dataset
        if noise:
            dataset_path = os.path.join(f"{datadir}/noise-{dist}-distance/dataset.pt")
        else:
            dataset_path = os.path.join(f"{datadir}/{dist}-distance/dataset.pt")
        dataset = torch.load(dataset_path)
        dataset = global_connect_feat_eng(
            dataset
        )  # Adding these original connectivity features with global connection should allow the model to learn the chiral information.
        print(f"Dataset contains {len(dataset)} graphs.")

        # Model
        modelname = f"global_e3nn_feateng"
        if noise:
            log_dir = f"logs/noise-{dist}-distance-{modelname}"
        else:
            log_dir = f"logs/{dist}-distance-{modelname}"
        model_args["layers"] = (
            4  # 4 is enough to propagate chirality information with global connections
        )
        model_args["max_radius"] = 2.5 * (dist + 1)
        model = CustomNetwork(
            irreps_in=o3.Irreps(model_args["irreps_in"]),
            irreps_hidden=o3.Irreps(model_args["irreps_hidden"]),
            irreps_out=o3.Irreps(model_args["irreps_out"]),
            irreps_node_attr=o3.Irreps(model_args["irreps_node_attr"]),
            irreps_edge_attr=o3.Irreps(model_args["irreps_edge_attr"]),
            layers=model_args["layers"],
            max_radius=model_args["max_radius"],
            number_of_basis=model_args["number_of_basis"],
            radial_layers=model_args["radial_layers"],
            radial_neurons=model_args["radial_neurons"],
            num_neighbors=model_args["num_neighbors"],
            num_nodes=model_args["num_nodes"],
            reduce_output=model_args["reduce_output"],
            num_classes=model_args["num_classes"],
            num_heads=model_args["num_heads"],
        ).to(device)
        print(f"Training model for distance {dist}...")
        train_val_test_model(model, model_args, dataset, dist, log_dir)

    print("Experiment complete.")


if __name__ == "__main__":
    main()
