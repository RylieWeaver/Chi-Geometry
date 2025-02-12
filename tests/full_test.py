# General
import os
import pytest

# Torch
import torch
from torch_geometric.loader import DataLoader

# E3NN
from e3nn import o3

# Custom
from chi_geometry import load_dataset_json, create_dataset
from examples.model import Network, load_model_json, train
from examples.model.train import (
    test as model_test,
)  # Necessary to avoid pytest thinking this function is a test


def run_model(dataset, model_args):
    # Setup
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Shuffle and split dataset
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(dataset))
    shuffled_dataset = [dataset[i] for i in shuffled_indices]

    train_size = int(0.8 * len(shuffled_dataset))
    train_dataset = shuffled_dataset[:train_size]
    test_dataset = shuffled_dataset[train_size:]

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=model_args["batch_size"], shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=model_args["batch_size"])

    # Initialize the model, optimizer, and loss function
    num_classes = model_args["num_classes"]
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
        num_classes=num_classes,
        num_heads=model_args["num_heads"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_args["lr"])
    # Define loss
    class_weights = torch.tensor(model_args["class_weights"], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(1, model_args["epochs"] + 1):
        loss = train(model, train_loader, optimizer, criterion, device)
        _, train_accuracies = model_test(
            model, train_loader, criterion, device, num_classes
        )
        _, test_accuracies = model_test(
            model, test_loader, criterion, device, num_classes
        )

        # Format accuracies
        train_acc_str = ", ".join(
            [f"Class {i}: {acc:.2f}" for i, acc in enumerate(train_accuracies)]
        )
        test_acc_str = ", ".join(
            [f"Class {i}: {acc:.2f}" for i, acc in enumerate(test_accuracies)]
        )

        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: [{train_acc_str}], Test Acc: [{test_acc_str}]"
        )

    print("Training complete.")


@pytest.mark.parametrize("ctype", ["simple", "crossed", "classic"])
@pytest.mark.parametrize("distance", [1, 2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("noise", [True, False])
def test_run_model(ctype, distance, noise):
    # Read Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_config_path = os.path.join(script_dir, "test_dataset_config.json")
    model_config_path = os.path.join(script_dir, "test_model_config.json")
    dataset_args = load_dataset_json(dataset_config_path)
    model_args = load_model_json(model_config_path)
    save_path = os.path.join(
        script_dir, f"datasets/dataset_ctype-{ctype}_dist-{distance}_noise-{noise}.pt"
    )

    # Create or load dataset
    if not os.path.exists(save_path):
        dataset = create_dataset(
            num_samples=dataset_args["num_samples"],
            type=ctype,
            chirality_distance=distance,
            species_range=dataset_args["species_range"],
            points=dataset_args["points"],
            noise=noise,
        )
        if os.path.dirname(save_path):  # Check if there is a directory in the path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dataset, save_path)
        print(f"Dataset saved as {save_path}")
    else:
        dataset = torch.load(save_path, weights_only=False)
        print(f"Dataset loaded from {save_path}")
    print(f"Dataset contains {len(dataset)} graphs.")

    # Run model
    run_model(dataset, model_args)
