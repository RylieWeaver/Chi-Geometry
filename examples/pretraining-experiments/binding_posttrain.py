# General
import os
import json
import datetime
import argparse
from tqdm import tqdm
import random

# Torch
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_scatter import scatter

# E3NN
from e3nn import o3

# Chi-Geometry
from chi_geometry.dataset import load_dataset_json
from chi_geometry.model import Network, load_model_json, process_batch
from experiment_utils.model import Wrapped_Network
from experiment_utils.train_val_test import train_val_test_model_no_accuracies


def load_model_pretrained(device, pretrained_dir=None):
    # Choose model args
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if pretrained_dir is not None:
        model_config_path = os.path.join(pretrained_dir, "model_args.json")
    else:
        model_config_path = os.path.join(script_dir, "model_config.json")
    model_args = load_model_json(model_config_path)
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    # Load model
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
        num_neighbors=model_args["num_neighbors"],
        num_nodes=model_args["num_nodes"],
        reduce_output=model_args["reduce_output"],
        num_classes=model_args["num_classes"],
        num_heads=model_args["num_heads"],
    ).to(device)

    # Load pretrained weights
    if pretrained_dir is not None:
        print(f"Loading pretrained weights from: {pretrained_dir}")
        base.load_state_dict(
            torch.load(
                os.path.join(pretrained_dir, "best_model.pt"), map_location=device
            ),
            strict=False,
        )

    # Wrap the pretrained network for the binding affinity task
    model = Wrapped_Network(base, model_args["num_classes"], 1).to(device)

    # Choose modelname
    if pretrained_dir is not None:
        modelname = f"from-pretrained_local_e3nn"
    else:
        modelname = f"from-scratch_local_e3nn"

    return model, model_args, modelname


def main():
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load binding affinity dataset
    datadir = os.path.join(script_dir, "binding_affinity_data")
    train_dataset = torch.load(os.path.join(datadir, "train.pt"))
    val_dataset = torch.load(os.path.join(datadir, "val.pt"))
    test_dataset = torch.load(os.path.join(datadir, "test.pt"))
    # Shuffle subsetted datasets
    torch.manual_seed(42)
    random.seed(42)
    random.shuffle(train_dataset)
    random.shuffle(val_dataset)

    # Args
    pretrained_dir = f"logs/pretrained_local_e3nn"
    criterion = torch.nn.MSELoss()

    # Run repeatedly for various dataset size
    for dataset_size in [100, 316, 1000, 3162, 10000]:

        # Get dataset
        train = train_dataset[:dataset_size]
        val = val_dataset[: int(0.125 * dataset_size)]
        test = test_dataset
        print(
            f"SAMPLE NUMBERS: Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
        )

        # Train models repeatedly
        for repetition in range(5):
            # Train a model from scratch
            print(f"Training model from scratch...")
            model, model_args, modelname = load_model_pretrained(None)
            log_dir = f"logs/binding_affinity-{modelname}-{dataset_size}-samples-repetition-{repetition+1}"
            train_val_test_model_no_accuracies(
                model,
                model_args,
                train,
                val,
                test,
                criterion,
                log_dir,
            )
            # Train a model from pretrained
            print(f"Training model from pretrained with dataset size {dataset_size}...")
            model, model_args, modelname = load_model_pretrained(pretrained_dir)
            train_val_test_model_no_accuracies(
                model,
                model_args,
                train,
                val,
                test,
                criterion,
                log_dir,
            )

    print("Experiment complete.")


if __name__ == "__main__":
    main()
