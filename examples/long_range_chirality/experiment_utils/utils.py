# General
import os
import json
import random
import numpy as np

# Torch
import torch

# Chi-Geometry
from chi_geometry import load_dataset_json, create_dataset
from examples.utils import shuffle_split_dataset


def create_hop_distance_datasets(distances, config_path, datadir="datasets"):
    # Read Config
    dataset_args = load_dataset_json(config_path)
    num_samples = dataset_args["num_samples"]
    # Make datasets and save configs
    for dist in distances:
        # Set randomness
        seed = 42 + dist
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Overwrite args
        dataset_args["chirality_distance"] = dist

        # Deterministic datasets
        dataset_args["noise"] = False
        dataset = create_dataset(
            num_samples=num_samples,
            chirality_type=dataset_args["chirality_type"],
            chirality_distance=dist,
            species_range=dataset_args["species_range"],
            noise=dataset_args["noise"],
        )
        train_dataset, val_dataset, test_dataset = shuffle_split_dataset(dataset, seed)
        # Save
        save_path = os.path.join(f"{datadir}/{dist}-distance")
        os.makedirs(save_path, exist_ok=True)  # Make dir if needed
        torch.save(train_dataset, os.path.join(f"{save_path}/train.pt"))
        torch.save(val_dataset, os.path.join(f"{save_path}/val.pt"))
        torch.save(test_dataset, os.path.join(f"{save_path}/test.pt"))
        print(f"Dataset saved at {save_path}")

        with open(
            os.path.join(f"{datadir}/{dist}-distance/dataset_config.json"), "w"
        ) as f:
            json.dump(dataset_args, f)

        # Noisy dataset
        dataset_args["noise"] = False
        dataset = create_dataset(
            num_samples=num_samples,
            chirality_type=dataset_args["chirality_type"],
            chirality_distance=dist,
            species_range=dataset_args["species_range"],
            noise=dataset_args["noise"],
        )
        train_dataset, val_dataset, test_dataset = shuffle_split_dataset(dataset, seed)
        # Save
        save_path = os.path.join(f"{datadir}/noise-{dist}-distance")
        os.makedirs(save_path, exist_ok=True)  # Make dir if needed
        torch.save(train_dataset, os.path.join(f"{save_path}/train.pt"))
        torch.save(val_dataset, os.path.join(f"{save_path}/val.pt"))
        torch.save(test_dataset, os.path.join(f"{save_path}/test.pt"))
        print(f"Dataset saved at {save_path}")
        with open(
            os.path.join(f"{datadir}/noise-{dist}-distance/dataset_config.json"), "w"
        ) as f:
            json.dump(dataset_args, f)
