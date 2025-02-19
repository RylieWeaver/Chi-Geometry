# General
import os
import json

# Torch
import torch

# Chi-Geometry
from chi_geometry import load_dataset_json, create_dataset


def create_hop_distance_datasets(distances, config_path, datadir="datasets"):
    # Read Config
    dataset_args = load_dataset_json(config_path)
    num_samples = dataset_args["num_samples"]
    # Make datasets and save configs
    for dist in distances:
        dataset_args["chirality_distance"] = dist

        # Deterministic dataset
        dataset = create_dataset(
            num_samples=num_samples,
            chirality_type=dataset_args["chirality_type"],
            chirality_distance=dist,
            species_range=dataset_args["species_range"],
            noise=False,
        )
        # Save
        save_path = os.path.join(f"{datadir}/{dist}-distance/dataset.pt")
        if os.path.dirname(save_path):  # Check if there is a directory in the path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dataset, save_path)
        print(f"Dataset saved as {save_path}")
        dataset_args["noise"] = False
        with open(
            os.path.join(f"{datadir}/{dist}-distance/dataset_config.json"), "w"
        ) as f:
            json.dump(dataset_args, f)

        # Noisy dataset
        dataset = create_dataset(
            num_samples=num_samples,
            chirality_type=dataset_args["chirality_type"],
            chirality_distance=dist,
            species_range=dataset_args["species_range"],
            noise=True,
        )
        # Save
        save_path = os.path.join(f"{datadir}/noise-{dist}-distance/dataset.pt")
        if os.path.dirname(save_path):  # Check if there is a directory in the path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dataset, save_path)
        print(f"Dataset saved as {save_path}")
        dataset_args["noise"] = True
        with open(
            os.path.join(f"{datadir}/noise-{dist}-distance/dataset_config.json"), "w"
        ) as f:
            json.dump(dataset_args, f)
