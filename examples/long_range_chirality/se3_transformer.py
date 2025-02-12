# General
import os
import datetime

# Torch
import torch

# Chi-Geometry
from examples.model import load_model_json
from experiment_utils.utils import make_global_connections, create_all_datasets
from experiment_utils.model import WrappedEquiformer
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
    noise = True
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

        # Model
        modelname = f"se(3)_transformer"
        if noise:
            log_dir = f"logs/noise-{dist}-distance-{modelname}"
        else:
            log_dir = f"logs/{dist}-distance-{modelname}"
        model_args["layers"] = (
            4  # 4 is enough to propagate chirality information with global connections
        )
        model = WrappedEquiformer(
            input_dim=model_args["input_dim"],
            hidden_dim=model_args["hidden_dim"],
            layers=model_args["layers"],
            num_classes=model_args["num_classes"],
        ).to(device)
        print(f"Training model for distance {dist}...")
        train_val_test_model(model, model_args, dataset, dist, log_dir)

    print("Experiment complete.")


if __name__ == "__main__":
    main()
