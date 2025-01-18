# General
import os
import json
import datetime
import random

# Torch
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# E3NN
from e3nn import o3

# Custom
from chi_geometry.model import Network, load_model_json
from experiment_utils.train_val_test import train_val_test_model_accuracies


def main():
    # Setup
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    distances = [1, 2, 3]

    # Create datasets
    # dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    # create_all_datasets(distances, dataset_config_path)

    # Args
    model_config_path = os.path.join(script_dir, "model_config.json")
    model_args = load_model_json(model_config_path)
    noise = False
    datadir = "datasets"
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )
    ## Increase layers with distance to ensure propagation of chiral information
    model_args["layers"] = max(distances) + 3
    ## Higher distances will have a lower proportion of chiral centers, so we weight chiral classifications at higher distances more
    num_classes = model_args["num_classes"]
    shift_weights = [0.0] + [num_classes * sum(distances) / len(distances)] * (
        num_classes - 1
    )
    class_weights = list(map(sum, zip(model_args["class_weights"], shift_weights)))
    class_weights = torch.tensor(class_weights, device=device)

    # Gather Composite Dataset
    dataset = []
    for dist in distances:
        if noise:
            dataset_path = os.path.join(f"{datadir}/noise-{dist}-distance/dataset.pt")
        else:
            dataset_path = os.path.join(f"{datadir}/{dist}-distance/dataset.pt")
        dataset.extend(torch.load(dataset_path))

    # Shuffle and split dataset
    torch.manual_seed(42)
    random.seed(42)
    random.shuffle(dataset)
    train_dataset = dataset[: int(0.8 * len(dataset))]
    val_dataset = dataset[int(0.8 * len(dataset)) : int(0.9 * len(dataset))]
    test_dataset = dataset[int(0.9 * len(dataset)) :]
    print(
        f"SAMPLE NUMBERS: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Model
    log_dir = f"logs/pretrained_local_e3nn"
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
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
        num_neighbors=model_args["num_neighbors"],
        num_nodes=model_args["num_nodes"],
        reduce_output=model_args["reduce_output"],
        num_classes=model_args["num_classes"],
        num_heads=model_args["num_heads"],
    ).to(device)
    print(f"Pre-Training model for distances {distances} and noise {noise}...")
    train_val_test_model_accuracies(
        model,
        model_args,
        train_dataset,
        val_dataset,
        test_dataset,
        criterion,
        num_classes,
        log_dir,
    )

    print("Experiment complete.")


if __name__ == "__main__":
    main()
