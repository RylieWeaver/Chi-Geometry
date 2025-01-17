# General
import os
import json
from typing import Dict, Optional
import networkx as nx

# Torch
import torch

# Chi-Geometry
from chi_geometry.dataset import load_dataset_json, create_dataset


def create_all_datasets(distances, config_path, datadir="datasets"):
    # Read Config
    dataset_args = load_dataset_json(config_path)
    num_samples = dataset_args["num_samples"]
    # Make datasets and save configs
    for dist in distances:
        dataset_args["distance"] = dist
        # Deterministic dataset
        create_dataset(
            num_samples=num_samples,
            type=dataset_args["type"],
            chirality_distance=dist,
            species_range=dataset_args["species_range"],
            points=dataset_args["points"],
            save_path=os.path.join(f"{datadir}/{dist}-distance/dataset.pt"),
            noise=False,
        )
        dataset_args["noise"] = False
        with open(
            os.path.join(f"{datadir}/{dist}-distance/dataset_config.json"), "w"
        ) as f:
            json.dump(dataset_args, f)
        # Noisy dataset
        create_dataset(
            num_samples=num_samples,
            type=dataset_args["type"],
            chirality_distance=dist,
            species_range=dataset_args["species_range"],
            points=dataset_args["points"],
            save_path=os.path.join(f"{datadir}/noisy-{dist}-distance/dataset.pt"),
            noise=True,
        )
        dataset_args["noise"] = True
        with open(
            os.path.join(f"{datadir}/noisy-{dist}-distance/dataset_config.json"), "w"
        ) as f:
            json.dump(dataset_args, f)


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


def make_bfs_hop_edge_attr(dataset):
    """
    For each edge, store the BFS-based hop distance (shortest path in the graph).
    *Potentially expensive* for large graphs, as we do BFS from each node.
    """

    for data in dataset:
        edge_index = data.edge_index.cpu().numpy()
        num_nodes = data.pos.size(0)

        # Construct an undirected graph in NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, dst)

        # Compute shortest_path_length for all pairs via repeated BFS
        all_dist = dict(nx.all_pairs_shortest_path_length(G))

        # Construct BFS-hop distance for each edge
        num_edges = edge_index.shape[1]
        hop_distance = []
        for i in range(num_edges):
            src, dst = edge_index[:, i]
            dist = all_dist[src][dst]  # BFS distance
            hop_distance.append(dist)

        hop_distance = torch.tensor(
            hop_distance, dtype=torch.float32, device=data.edge_index.device
        ).unsqueeze(1)
        data.edge_attr = torch.cat([hop_distance], dim=1)

    return dataset
