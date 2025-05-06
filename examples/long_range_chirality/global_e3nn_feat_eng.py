# General
import os
import random
import numpy as np

# Torch
import torch

# E3NN
from e3nn import o3

# Chi-Geometry
from examples.utils import load_model_json
from examples.utils import (
    CustomNetwork,
    train_val_test_model_classification,
    shuffle_split_dataset,
    global_connect_feat_eng,
    get_avg_degree,
    get_avg_nodes,
    get_max_distance,
)
from experiment_utils.utils import create_hop_distance_datasets


def main():
    # Setup
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    distances = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Create datasets
    # dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    # create_hop_distance_datasets(distances, dataset_config_path)

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
        # Set randomness
        seed = 42 + dist
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Dataset
        if noise:
            dataset_path = os.path.join(f"{datadir}/noise-{dist}-distance/")
        else:
            dataset_path = os.path.join(f"{datadir}/{dist}-distance/")
        train_dataset = torch.load(os.path.join(f"{dataset_path}/train.pt"), weights_only=False)
        val_dataset = torch.load(os.path.join(f"{dataset_path}/val.pt"), weights_only=False)
        test_dataset = torch.load(os.path.join(f"{dataset_path}/test.pt"), weights_only=False)
        print(
            f"Datasets Loaded with Train: {len(train_dataset)}, "
            f"Val: {len(val_dataset)}, Test: {len(test_dataset)}\n"
        )

        # Feature Engineering
        for dataset in [train_dataset, val_dataset, test_dataset]:
            dataset = global_connect_feat_eng(
                dataset
            )  # Adding these original connectivity features with global connection should allow the model to learn the chiral information.

        # Get statistics
        avg_degree = get_avg_degree(train_dataset)
        max_radius = get_max_distance(train_dataset) * 1.01

        # Model
        modelname = f"global_e3nn_feateng"
        if noise:
            log_dir = f"logs/noise-{dist}-distance-{modelname}"
        else:
            log_dir = f"logs/{dist}-distance-{modelname}"
        model_args[
            "layers"
        ] = 4  # 4 is enough to propagate chirality information with global connections
        model = CustomNetwork(
            irreps_in=o3.Irreps(model_args["irreps_in"]),
            irreps_hidden=o3.Irreps(model_args["irreps_hidden"]),
            irreps_out=o3.Irreps(model_args["irreps_out"]),
            irreps_node_attr=o3.Irreps(model_args["irreps_node_attr"]),
            irreps_edge_attr=o3.Irreps(model_args["irreps_edge_attr"]),
            layers=model_args["layers"],
            max_radius=max_radius,
            number_of_basis=model_args["number_of_basis"],
            radial_layers=model_args["radial_layers"],
            radial_neurons=model_args["radial_neurons"],
            avg_degree=avg_degree,
            output_dim=model_args["output_dim"],
        ).to(device)
        print(f"Training model for distance {dist}...")
        # Higher distances will have a lower proportion of chiral centers, so we weight chiral classifications at higher distances more
        num_classes = model_args["output_dim"]
        shift_weights = [0.0] + [num_classes * dist] * (num_classes - 1)
        model_args["class_weights"] = list(
            map(sum, zip(model_args["class_weights"], shift_weights))
        )
        train_val_test_model_classification(
            model, model_args, train_dataset, val_dataset, test_dataset, log_dir
        )

    print("Experiment complete.")


if __name__ == "__main__":
    main()
