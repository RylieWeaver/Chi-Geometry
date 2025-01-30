# General
import os
import json
import datetime
import argparse
from tqdm import tqdm
import random
import gc

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

    return model, modelname


def main():
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_args = load_model_json(os.path.join(script_dir, "model_config.json"))
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    # Load binding affinity dataset
    datadir = os.path.join(script_dir, "binding_affinity_data")
    train_dataset = torch.load(os.path.join(datadir, "train.pt"))
    val_dataset = torch.load(os.path.join(datadir, "val.pt"))
    test_dataset = torch.load(os.path.join(datadir, "test.pt"))
    random.shuffle(train_dataset)
    random.shuffle(val_dataset)

    # Args
    pretrained_dir = f"logs/pretrained_local_e3nn-noisy_and_deterministic-2"
    criterion = torch.nn.MSELoss()

    # Run repeatedly for various dataset size
    # for dataset_size in [10, 18, 32, 56, 100, 178, 316, 562, 1000]:
    for dataset_size in [1000]:

        # Set up repetitions
        num_repetitions = 100
        val_size = int(0.125 * dataset_size)
        # We should have enough samples for both train and val repetitions without reusing samples to ensure bootstrap independence
        assert len(train_dataset) >= dataset_size * num_repetitions
        assert len(val_dataset) >= val_size * num_repetitions

        # Train models repeatedly
        for repetition in range(75, 100):

            # Get subset which will be used for both scratch and pretrained
            train = train_dataset[
                dataset_size * repetition : dataset_size * (repetition + 1)
            ]
            val = val_dataset[val_size * repetition : val_size * (repetition + 1)]
            test = test_dataset  # No subsetting or shuffling necessary for the test set
            print(
                f"SAMPLE NUMBERS: Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
            )

            # Set a seed to be consistent between each pair but different between repetitions
            seed = 42 + repetition
            random.seed(seed)
            torch.manual_seed(seed)
            python_state = random.getstate()
            torch_state = torch.get_rng_state()
            # Train a model from scratch
            print(f"Training model from scratch...")
            model, modelname = load_model_pretrained(device, None)
            log_dir = f"logs_full-trained2/binding_affinity-{modelname}-{dataset_size}-samples-repetition-{repetition+1}"
            train_val_test_model_no_accuracies(
                model,
                model_args,
                train,
                val,
                test,
                criterion,
                log_dir,
            )

            # Reset random state
            random.setstate(python_state)
            torch.set_rng_state(torch_state)
            # Train a model from pretrained
            print(f"Training model from pretrained with dataset size {dataset_size}...")
            model, modelname = load_model_pretrained(device, pretrained_dir)
            log_dir = f"logs_full-trained2/binding_affinity-{modelname}-{dataset_size}-samples-repetition-{repetition+1}"
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
