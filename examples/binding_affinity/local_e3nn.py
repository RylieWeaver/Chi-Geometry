# General
import os

# Torch
import torch

# E3NN
from e3nn import o3

# Chi-Geometry
from examples.utils import (
    load_model_json,
    WrappedModel,
    Network,
    train_val_test_model_regression,
    get_avg_degree,
    get_avg_nodes,
)


def main():
    # Setup
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    repetitions = 10

    # Args
    model_config_path = os.path.join(script_dir, "e3nn_local_model_config.json")
    model_args = load_model_json(model_config_path)
    datadir = "datasets"
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    # Load Data
    train_dataset = torch.load(os.path.join(f"{datadir}/train_largest.pt"))
    val_dataset = torch.load(os.path.join(f"{datadir}/val_largest.pt"))
    test_dataset = torch.load(os.path.join(f"{datadir}/test_largest.pt"))
    print(
        f"Datasets Loaded with Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}\n"
    )

    # Get statistics
    avg_degree = get_avg_degree(train_dataset)
    avg_nodes = get_avg_nodes(train_dataset)

    # Multiple repetitions
    for repetition in range(repetitions):
        # Args
        modelname = f"local_e3nn"
        model_args["max_radius"] = 5.0
        log_dir = f"logs/{modelname}_repetition_{repetition+1}"
        # Model
        base = Network(
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
            avg_degree=avg_degree,
            output_dim=model_args["hidden_dim"],
        ).to(device)
        model = WrappedModel(
            base, model_args["hidden_dim"], model_args["output_dim"], avg_nodes
        ).to(device)
        # Training
        print(f"Training model for repetition {repetition+1}...")
        train_val_test_model_regression(
            model, model_args, train_dataset, val_dataset, test_dataset, log_dir
        )
    print("Experiment complete.")


if __name__ == "__main__":
    main()
