# General
import os

# Torch
import torch

# Chi-Geometry
from examples.utils import (
    load_model_json,
    WrappedModel,
    Equiformer,
    train_val_test_model_regression,
    make_global_connections,
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
        dataset = make_global_connections(
            dataset
        )  # Adding these original connectivity features with global connection should allow the model to learn the chiral information.

    # Multiple repetitions
    for repetition in range(repetitions):
        # Args
        modelname = f"se3_transformer"
        model_args["max_radius"] = 20.0
        log_dir = f"logs/{modelname}_{repetition+1}"
        # Model
        base = Equiformer(
            input_dim=model_args["input_dim"],
            hidden_dim=model_args["hidden_dim"],
            layers=model_args["layers"],
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
