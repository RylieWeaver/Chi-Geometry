# General
import os
import datetime

# Torch
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# E3NN
from e3nn import o3

# Custom
from chi_geometry.dataset import load_dataset_json, create_dataset
from chi_geometry.model import Network, load_model_json, train, test


def create_all_datasets(distances):
    # Read Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    # Make arguments
    dataset_args = load_dataset_json(dataset_config_path)
    num_samples = dataset_args["num_samples"]
    # Make datasets
    for dist in distances:
        save_path = os.path.join(f"datasets/{num_samples}-samples_{dist}-distance.pt")
        create_dataset(
            num_samples=num_samples,
            type=dataset_args["type"],
            chirality_distance=dist,
            species_range=dataset_args["species_range"],
            points=dataset_args["points"],
            save_path=save_path,
        )


def train_model(distance):
    # Read Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    model_config_path = os.path.join(script_dir, "model_config.json")
    dataset_args = load_dataset_json(dataset_config_path)
    model_args = load_model_json(model_config_path)

    # Log Directory
    run_dir = os.path.join(script_dir, f"logs/{distance}-distance")
    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, "best_model.pt")
    training_log_path = os.path.join(run_dir, "training_log.txt")
    final_results_path = os.path.join(run_dir, "final_results.txt")

    # Device
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Dataset
    torch.manual_seed(42)
    dataset_path = os.path.join(
        f'datasets/{dataset_args["num_samples"]}-samples_{distance}-distance.pt'
    )
    dataset = torch.load(dataset_path)
    print(f"Dataset contains {len(dataset)} graphs.")
    shuffled_indices = torch.randperm(len(dataset))
    shuffled_dataset = [dataset[i] for i in shuffled_indices]

    train_size = int(0.8 * len(shuffled_dataset))
    val_size = int(0.1 * len(shuffled_dataset))
    train_dataset = shuffled_dataset[:train_size]
    val_dataset = shuffled_dataset[train_size : train_size + val_size]
    test_dataset = shuffled_dataset[train_size + val_size :]

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=model_args["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=model_args["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=model_args["batch_size"])

    # Initialize the model, optimizer, and loss function
    num_classes = model_args["num_classes"]
    layers = (
        distance + 3
    )  # Number of layers is always distance + 3 to ensure enough message-passing for chirality awareness
    model = Network(
        irreps_in=o3.Irreps(model_args["irreps_in"]),
        irreps_hidden=o3.Irreps(model_args["irreps_hidden"]),
        irreps_out=o3.Irreps(model_args["irreps_out"]),
        irreps_node_attr=o3.Irreps(model_args["irreps_node_attr"]),
        irreps_edge_attr=o3.Irreps(model_args["irreps_edge_attr"]),
        layers=layers,
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

    # Optimizing
    optimizer = torch.optim.Adam(model.parameters(), lr=model_args["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=25,
        verbose=True,
        min_lr=model_args["lr"] / 1e3,
    )

    # Early stopping parameters
    patience = 75
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Define loss
    ## Higher distances will have a lower proportion of chiral centers, so we weight chiral classifications at higher distances more
    class_weights = class_weights = list(
        map(sum, zip(model_args["class_weights"], [0.0, 3 * distance, 3 * distance]))
    )  # Weird logic to sum lists element-wise
    class_weights = torch.tensor(class_weights, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # -- Training loop with logging --
    with open(training_log_path, "w") as log_file:
        log_file.write(
            "epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc\n"
        )

        for epoch in range(1, model_args["epochs"] + 1):
            # Train for one epoch
            loss = train(model, train_loader, optimizer, criterion, device)

            # Compute stats on train / val / test
            train_loss, train_accuracies = test(
                model, train_loader, criterion, device, num_classes
            )
            val_loss, val_accuracies = test(
                model, val_loader, criterion, device, num_classes
            )
            test_loss, test_accuracies = test(
                model, test_loader, criterion, device, num_classes
            )

            # Format accuracies for printing
            train_acc_str = ", ".join(
                [f"Class {i}: {acc:.2f}" for i, acc in enumerate(train_accuracies)]
            )
            val_acc_str = ", ".join(
                [f"Class {i}: {acc:.2f}" for i, acc in enumerate(val_accuracies)]
            )
            test_acc_str = ", ".join(
                [f"Class {i}: {acc:.2f}" for i, acc in enumerate(test_accuracies)]
            )

            # Print to console
            print(f"\nEpoch: {epoch:03d}, During Training Loss: {loss:.4f}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: [{train_acc_str}]")
            print(f"Val Loss: {val_loss:.4f},   Val Acc:   [{val_acc_str}]")
            print(f"Test Loss: {test_loss:.4f}, Test Acc:  [{test_acc_str}]")

            # Write to the training log file
            avg_train_acc = sum(train_accuracies) / len(train_accuracies)
            avg_val_acc = sum(val_accuracies) / len(val_accuracies)
            avg_test_acc = sum(test_accuracies) / len(test_accuracies)
            log_file.write(
                f"{epoch},{train_loss:.4f},{avg_train_acc:.4f},"
                f"{val_loss:.4f},{avg_val_acc:.4f},"
                f"{test_loss:.4f},{avg_test_acc:.4f}\n"
            )
            log_file.flush()  # ensure data is written immediately

            # Step the learning rate scheduler with the current validation loss
            scheduler.step(val_loss)

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(
                    model.state_dict(), best_model_path
                )  # Save new best model state
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")

    ############################################################################
    # -- Use the best model for final evaluation --
    best_model = Network(
        irreps_in=o3.Irreps(model_args["irreps_in"]),
        irreps_hidden=o3.Irreps(model_args["irreps_hidden"]),
        irreps_out=o3.Irreps(model_args["irreps_out"]),
        irreps_node_attr=o3.Irreps(model_args["irreps_node_attr"]),
        irreps_edge_attr=o3.Irreps(model_args["irreps_edge_attr"]),
        layers=layers,
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

    best_model.load_state_dict(torch.load(best_model_path))

    # -- Final stats with the best model --
    final_train_loss, final_train_acc = test(
        best_model, train_loader, criterion, device, num_classes
    )
    final_val_loss, final_val_acc = test(
        best_model, val_loader, criterion, device, num_classes
    )
    final_test_loss, final_test_acc = test(
        best_model, test_loader, criterion, device, num_classes
    )

    # Format for printing
    train_acc_str = ", ".join(
        [f"Class {i}: {acc:.2f}" for i, acc in enumerate(final_train_acc)]
    )
    val_acc_str = ", ".join(
        [f"Class {i}: {acc:.2f}" for i, acc in enumerate(final_val_acc)]
    )
    test_acc_str = ", ".join(
        [f"Class {i}: {acc:.2f}" for i, acc in enumerate(final_test_acc)]
    )

    print("\n===== Final Stats (Best Model) =====")
    print(f"Train Loss: {final_train_loss:.4f}, Train Acc: [{train_acc_str}]")
    print(f"Val Loss:   {final_val_loss:.4f},   Val Acc:   [{val_acc_str}]")
    print(f"Test Loss:  {final_test_loss:.4f},  Test Acc:  [{test_acc_str}]")

    # -- Write final results to a file in the run_dir --
    with open(final_results_path, "w") as f:
        f.write("===== Final Stats (Best Model) =====\n")
        f.write(f"Train Loss: {final_train_loss:.4f}, Train Acc: [{train_acc_str}]\n")
        f.write(f"Val Loss:   {final_val_loss:.4f},   Val Acc:   [{val_acc_str}]\n")
        f.write(f"Test Loss:  {final_test_loss:.4f},  Test Acc:  [{test_acc_str}]\n")

    print(f"Evaluation complete. Results written to {final_results_path}\n\n")


def main():
    # Create datasets
    distances = [3, 4, 5, 6, 7, 8, 9, 10]
    # create_all_datasets(distances)

    # Train models
    for dist in distances:
        print(f"Training model for distance {dist}...")
        train_model(dist)

    print("Experiment complete.")


if __name__ == "__main__":
    main()