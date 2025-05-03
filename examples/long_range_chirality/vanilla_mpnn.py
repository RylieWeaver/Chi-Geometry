# General
import os
import random
import numpy as np

# Torch
import torch

# E3NN
from e3nn import o3

# Chi-Geometry
from examples.utils import (
    VanillaMPNN,
    load_model_json,
    train_val_test_model_classification,
    get_avg_degree,
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
    model_config_path = os.path.join(script_dir, "vanilla_mpnn_model_config.json")
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
            dataset_path = os.path.join(f"{datadir}/noise-{dist}-distance/")
        else:
            dataset_path = os.path.join(f"{datadir}/{dist}-distance/")
        train_dataset = torch.load(os.path.join(f"{dataset_path}/train.pt"))
        val_dataset = torch.load(os.path.join(f"{dataset_path}/val.pt"))
        test_dataset = torch.load(os.path.join(f"{dataset_path}/test.pt"))
        print(
            f"Datasets Loaded with Train: {len(train_dataset)}, "
            f"Val: {len(val_dataset)}, Test: {len(test_dataset)}\n"
        )

        # Get statistics
        avg_degree = get_avg_degree(train_dataset)

        # Higher distances will have a lower proportion of chiral centers, so we weight chiral classifications at higher distances more
        num_classes = model_args["output_dim"]
        shift_weights = [0.0] + [num_classes * dist] * (num_classes - 1)
        model_args["class_weights"] = list(
            map(sum, zip(model_args["class_weights"], shift_weights))
        )
        print(f"Training models for distance {dist}...")

        # Train repeated models to avg accuracy
        for repetition in range(repetitions):
            # Set randomness
            seed = 42 + repetition
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Model
            modelname = f"vanilla_mpnn"
            if noise:
                log_dir = (
                    f"logs/noise-{dist}-distance-{modelname}-repetition-{repetition+1}"
                )
            else:
                log_dir = f"logs/{dist}-distance-{modelname}-repetition-{repetition+1}"
            model_args[
                "layers"
            ] = 4  # 4 is enough to propagate chirality information with global connections
            model = VanillaMPNN(
                hidden_dim=model_args["hidden_dim"],
                layers=model_args["layers"],
                avg_degree=avg_degree,
                output_dim=model_args["output_dim"],
            ).to(device)
            train_val_test_model_classification(
                model, model_args, train_dataset, val_dataset, test_dataset, log_dir
            )
            print(f"Training complete for repetition {repetition+1}.")

    print("Experiment complete.")


if __name__ == "__main__":
    main()
