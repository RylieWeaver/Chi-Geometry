# General
import os

# Torch
import torch
from torch_geometric.loader import DataLoader

# E3NN
from e3nn import o3

# Custom
from chi_geometry.dataset import load_dataset_json, create_dataset
from chi_geometry.model import Network, load_model_json, train, test


def main():
    # Read Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    model_config_path = os.path.join(script_dir, "model_config.json")
    dataset_args = load_dataset_json(dataset_config_path)
    model_args = load_model_json(model_config_path)

    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load dataset
    if not os.path.exists(dataset_args["save_path"]):
        create_dataset(
            num_samples=dataset_args["num_samples"],
            type=dataset_args["type"],
            chirality_distance=dataset_args["chirality_distance"],
            species_range=dataset_args["species_range"],
            points=dataset_args["points"],
            save_path=dataset_args["save_path"],
        )
    dataset = torch.load(dataset_args["save_path"], weights_only=False)
    print(f"Dataset contains {len(dataset)} graphs.")

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
        _, train_accuracies = test(model, train_loader, criterion, device, num_classes)
        _, test_accuracies = test(model, test_loader, criterion, device, num_classes)

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


if __name__ == "__main__":
    main()
