# General
import os
import datetime

# Torch
import torch

# E3NN
from e3nn import o3

# Chi-Geometry
from examples.utils import (
    Network,
    load_model_json,
    train_val_test_model_classification,
    make_global_connections,
)
from experiment_utils.utils import create_hop_distance_datasets


def main():
    # Setup
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    distances = [1]
    repetitions = 10

    # Create datasets
    # dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    # create_hop_distance_datasets(distances, dataset_config_path)

    # Args
    model_config_path = os.path.join(script_dir, "e3nn_global_model_config.json")
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
        dataset = make_global_connections(dataset)
        print(f"Dataset contains {len(dataset)} graphs.")

        # Higher distances will have a lower proportion of chiral centers, so we weight chiral classifications at higher distances more
        num_classes = model_args["output_dim"]
        shift_weights = [0.0] + [num_classes * dist] * (num_classes - 1)
        class_weights = list(map(sum, zip(model_args["class_weights"], shift_weights)))
        class_weights = torch.tensor(class_weights, device=device)
        print(f"Training models for distance {dist}...")

        # Train repeated models to avg accuracy
        for repetition in range(repetitions):
            # Model
            modelname = f"global_e3nn"
            if noise:
                log_dir = (
                    f"logs/noise-{dist}-distance-{modelname}-repetition-{repetition+1}"
                )
            else:
                log_dir = f"logs/{dist}-distance-{modelname}-repetition-{repetition+1}"
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
                output_dim=model_args["output_dim"],
            ).to(device)
            train_val_test_model_classification(model, model_args, dataset, log_dir)
            print(f"Training complete for repetition {repetition+1}.")

    print("Experiment complete.")


if __name__ == "__main__":
    main()
