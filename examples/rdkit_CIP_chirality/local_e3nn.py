# General
import os
import datetime

# Torch
import torch

# E3NN
from e3nn import o3

# Chi-Geometry
from examples.utils import Network, load_model_json, train_val_test_model_classification


def main():
    # Setup
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    # Args
    model_config_path = os.path.join(script_dir, "e3nn_local_model_config.json")
    model_args = load_model_json(model_config_path)
    datadir = "datasets"
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    # Load Data
    train_dataset = torch.load(os.path.join(f"{datadir}/train_processed_sample.pt"))
    val_dataset = torch.load(os.path.join(f"{datadir}/val_processed_sample.pt"))
    test_dataset = torch.load(os.path.join(f"{datadir}/test_processed_sample.pt"))
    print(
        f"Datasets Loaded with Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}\n"
    )

    # Save datasets
    # torch.save(train_dataset[:8000], os.path.join(f"{datadir}/train_processed_sample.pt"))
    # torch.save(val_dataset[:1000], os.path.join(f"{datadir}/val_processed_sample.pt"))
    # torch.save(test_dataset[:1000], os.path.join(f"{datadir}/test_processed_sample.pt"))

    # Model
    modelname = f"local_e3nn"
    log_dir = f"logs/{modelname}"
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
    print(f"Training model...")
    train_val_test_model_classification(
        model, model_args, train_dataset, val_dataset, test_dataset, log_dir
    )

    print("Experiment complete.")


if __name__ == "__main__":
    main()
