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
        dataset_args["chirality_distance"] = dist
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
            save_path=os.path.join(f"{datadir}/noise-{dist}-distance/dataset.pt"),
            noise=True,
        )
        dataset_args["noise"] = True
        with open(
            os.path.join(f"{datadir}/noise-{dist}-distance/dataset_config.json"), "w"
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
