# General
import os
import json
import numpy as np

# Torch
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Custom
from examples.utils import train, test_classification, test_regression


def train_val_test_model_classification(
    model, model_args, train_dataset, val_dataset, test_dataset, log_dir
):
    # Log Directory
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "model_config.json"), "w") as f:
        json.dump(model_args, f)
    best_model_path = os.path.join(log_dir, "best_model.pt")
    training_log_path = os.path.join(log_dir, "training_log.txt")
    final_results_path = os.path.join(log_dir, "final_results.txt")

    # Device
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=model_args["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=model_args["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=model_args["batch_size"])

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
    num_classes = model_args["output_dim"]
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            model_args["class_weights"], dtype=torch.float32, device=device
        )
    )

    # -- Training loop with logging --
    with open(training_log_path, "w") as log_file:
        log_file.write(
            "epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc\n"
        )

        for epoch in range(1, model_args["epochs"] + 1):
            # Train for one epoch
            loss = train(model, train_loader, optimizer, criterion, device)

            # Compute stats on train / val / test
            train_loss, train_accuracies = test_classification(
                model, train_loader, criterion, device, num_classes
            )
            val_loss, val_accuracies = test_classification(
                model, val_loader, criterion, device, num_classes
            )
            test_loss, test_accuracies = test_classification(
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
            log_file.write(
                f"\nEpoch: {epoch:03d}, During Training Loss: {loss:.4f}\n"
                f"Train Loss: {train_loss:.4f}, Train Acc: [{train_acc_str}]\n"
                f"Val Loss: {val_loss:.4f},   Val Acc:   [{val_acc_str}]\n"
                f"Test Loss: {test_loss:.4f}, Test Acc:  [{test_acc_str}]\n"
            )
            log_file.flush()  # ensure data is written immediately

            # Step the learning rate scheduler with the current validation loss
            scheduler.step(val_loss)

            # Early Stopping Logic
            if val_loss+1e-9 < best_val_loss:
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
    model.load_state_dict(torch.load(best_model_path))

    # -- Final stats with the best model --
    final_train_loss, final_train_acc = test_classification(
        model, train_loader, criterion, device, num_classes
    )
    final_val_loss, final_val_acc = test_classification(
        model, val_loader, criterion, device, num_classes
    )
    final_test_loss, final_test_acc = test_classification(
        model, test_loader, criterion, device, num_classes
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


def train_val_test_model_regression(
    model, model_args, train_dataset, val_dataset, test_dataset, log_dir
):
    # Log Directory
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "model_config.json"), "w") as f:
        json.dump(model_args, f)
    best_model_path = os.path.join(log_dir, "best_model.pt")
    training_log_path = os.path.join(log_dir, "training_log.txt")
    final_results_path = os.path.join(log_dir, "final_results.txt")

    # Device
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=model_args["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=model_args["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=model_args["batch_size"])

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
    criterion = torch.nn.MSELoss()

    # -- Training loop with logging --
    with open(training_log_path, "w") as log_file:
        log_file.write("epoch,train_loss,val_loss,test_loss\n")

        for epoch in range(1, model_args["epochs"] + 1):
            # Train for one epoch
            loss = train(model, train_loader, optimizer, criterion, device)

            # Compute stats on train / val / test
            train_loss = test_regression(model, train_loader, criterion, device)
            val_loss = test_regression(model, val_loader, criterion, device)
            test_loss = test_regression(model, test_loader, criterion, device)

            # Print to console
            print(f"\nEpoch: {epoch:03d}, During Training Loss: {loss:.4f}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}")

            # Write to the training log file
            log_file.write(
                f"{epoch},{train_loss:.4f}," f"{val_loss:.4f}," f"{test_loss:.4f}\n"
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
    model.load_state_dict(torch.load(best_model_path))

    # -- Final stats with the best model --
    final_train_loss = test_regression(model, train_loader, criterion, device)
    final_val_loss = test_regression(model, val_loader, criterion, device)
    final_test_loss = test_regression(model, test_loader, criterion, device)

    print("\n===== Final Stats (Best Model) =====")
    print(f"Train Loss: {final_train_loss:.4f}")
    print(f"Val Loss:   {final_val_loss:.4f}")
    print(f"Test Loss:  {final_test_loss:.4f}")

    # -- Write final results to a file in the run_dir --
    with open(final_results_path, "w") as f:
        f.write("===== Final Stats (Best Model) =====\n")
        f.write(f"Train Loss: {final_train_loss:.4f}\n")
        f.write(f"Val Loss:   {final_val_loss:.4f}\n")
        f.write(f"Test Loss:  {final_test_loss:.4f}\n")

    print(f"Evaluation complete. Results written to {final_results_path}\n\n")
