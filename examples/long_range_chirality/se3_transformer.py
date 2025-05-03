# General
import os
import random
import numpy as np

# Torch
import torch

# Chi-Geometry
from examples.utils import (
    load_model_json,
    Equiformer,
    make_global_connections,
    train_val_test_model_classification,
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
    model_config_path = os.path.join(script_dir, "se3_transformer_model_config.json")
    model_args = load_model_json(model_config_path)
    noise = True
    datadir = "datasets"
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    # Train models on each dataset
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

        # Higher distances will have a lower proportion of chiral centers, so we weight chiral classifications at higher distances more
        num_classes = model_args["output_dim"]
        shift_weights = [0.0] + [num_classes * dist] * (num_classes - 1)
        model_args["class_weights"] = list(
            map(sum, zip(model_args["class_weights"], shift_weights))
        )
        print(f"Training models for distance {dist}...")

        # Train repeated models to avg accuracy
        for repetition in range(repetitions):
            # Model
            modelname = f"se3_transformer"
            if noise:
                log_dir = (
                    f"logs/noise-{dist}-distance-{modelname}-repetition-{repetition+1}"
                )
            else:
                log_dir = f"logs/{dist}-distance-{modelname}-repetition-{repetition+1}"
            model_args[
                "layers"
            ] = 4  # 4 is enough to propagate chirality information with global connections
            model = Equiformer(
                input_dim=model_args["input_dim"],
                hidden_dim=model_args["hidden_dim"],
                layers=model_args["layers"],
                output_dim=model_args["output_dim"],
            ).to(device)
            train_val_test_model_classification(
                model, model_args, train_dataset, val_dataset, test_dataset, log_dir
            )
            print(f"Training complete for repetition {repetition+1}\n")

    print("Experiment complete.")


if __name__ == "__main__":
    main()
