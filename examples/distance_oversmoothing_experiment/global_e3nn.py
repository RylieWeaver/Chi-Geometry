# General
import os
import datetime

# Torch
import torch

# E3NN
from e3nn import o3

# Chi-Geometry
from chi_geometry.model import Network, load_model_json
from experiment_utils.utils import create_all_datasets
from experiment_utils.train_val_test import train_val_test_model


def main():
    # Setup
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    distances = [1]

    # Create datasets
    # dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    # create_all_datasets(distances, dataset_config_path)

    # Args
    model_config_path = os.path.join(script_dir, "model_config.json")
    model_args = load_model_json(model_config_path)
    noise = True  # This is crucial to showing the flaws in globally connected models, since it removes most distance patterns that dictate which is the chiral center.
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
        print(f"Dataset contains {len(dataset)} graphs.")

        # Model
        modelname = f"global_e3nn"
        if noise:
            log_dir = f"logs/noise-{dist}-distance-{modelname}"
        else:
            log_dir = f"logs/{dist}-distance-{modelname}"
        model_args["layers"] = (
            4  # 4 is enough to propagate chirality information with global connections
        )
        model = Network(
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