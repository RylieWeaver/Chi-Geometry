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
    CustomNetwork,
    train_val_test_model_regression,
    global_connect_feat_eng,
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

    # Feature Engineering
    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset = global_connect_feat_eng(
            dataset
        )  # Adding these original connectivity features with global connection should allow the model to learn the chiral information.

    # Multiple repetitions
    for repetition in range(repetitions):
        # Args
        modelname = f"global_e3nn_feateng"
        model_args["max_radius"] = 20.0
        log_dir = f"logs/{modelname}_{repetition+1}"
        # Model
        base = CustomNetwork(
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
            output_dim=model_args["hidden_dim"],
        ).to(device)
        model = WrappedModel(
            base, model_args["hidden_dim"], model_args["output_dim"]
        ).to(device)
        # Training
        print(f"Training model for repetition {repetition+1}...")
        train_val_test_model_regression(
            model, model_args, train_dataset, val_dataset, test_dataset, log_dir
        )
    print("Experiment complete.")


if __name__ == "__main__":
    main()
