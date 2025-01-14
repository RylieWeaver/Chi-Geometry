# General
import os
import json
import datetime
import argparse
from tqdm import tqdm

# Torch
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_scatter import scatter

# E3NN
from e3nn import o3

# Custom
from chi_geometry.dataset import load_dataset_json, create_dataset
from chi_geometry.model import Network, load_model_json, process_batch


def train_model(pretrained_dir=None, dataset_size=None):
    # Read Config
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Binding affinity dataset
    torch.manual_seed(42)
    datadir = "./binding_affinity_data"
    train_dataset = torch.load(os.path.join(datadir, "train.pt"))
    val_dataset = torch.load(os.path.join(datadir, "val.pt"))
    test_dataset = torch.load(os.path.join(datadir, "test.pt"))
    if dataset_size is not None:
        train_dataset = train_dataset[: int(0.8 * dataset_size)]
        val_dataset = val_dataset[: int(0.1 * dataset_size)]
        test_dataset = test_dataset[: int(0.1 * dataset_size)]
    else:
        dataset_size = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(
        f"SAMPLE NUMBERS: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Pretrained or not
    if pretrained_dir is not None:
        rundir_str = f"logs/binding_affinity_from_pretrained_{dataset_size}-fine-tune-samples_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        pretrained_model_config_path = os.path.join(pretrained_dir, "model_args.json")
        pretrained_model_args = load_model_json(pretrained_model_config_path)
    else:
        rundir_str = f"logs/binding_affinity_{dataset_size}-fine-tune-samples_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_config_path = os.path.join(script_dir, "model_config.json")
    model_args = load_model_json(model_config_path)

    # Log training details
    run_dir = os.path.join(script_dir, rundir_str)
    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, "best_model.pt")
    training_log_path = os.path.join(run_dir, "training_log.txt")
    final_results_path = os.path.join(run_dir, "final_results.txt")

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

    # Initialize the model, optimizer, and loss function
    if pretrained_dir is not None:
        # Create the same model architecture
        base = Network(
            irreps_in=o3.Irreps(pretrained_model_args["irreps_in"]),
            irreps_hidden=o3.Irreps(pretrained_model_args["irreps_hidden"]),
            irreps_out=o3.Irreps(pretrained_model_args["irreps_out"]),
            irreps_node_attr=o3.Irreps(pretrained_model_args["irreps_node_attr"]),
            irreps_edge_attr=o3.Irreps(pretrained_model_args["irreps_edge_attr"]),
            layers=pretrained_model_args["layers"],
            max_radius=pretrained_model_args["max_radius"],
            number_of_basis=pretrained_model_args["number_of_basis"],
            radial_layers=pretrained_model_args["radial_layers"],
            radial_neurons=pretrained_model_args["radial_neurons"],
            num_neighbors=pretrained_model_args["num_neighbors"],
            num_nodes=pretrained_model_args["num_nodes"],
            reduce_output=pretrained_model_args["reduce_output"],
            num_classes=pretrained_model_args["num_classes"],
            num_heads=pretrained_model_args["num_heads"],
        ).to(device)
        # Load pretrained weights
        print(f"Loading pretrained weights from: {pretrained_dir}")
        pretrained_model_path = os.path.join(pretrained_dir, "best_model.pt")
        state_dict = torch.load(pretrained_model_path, map_location=device)
        base.load_state_dict(state_dict, strict=False)
        # Wrap the model with graph agg and a linear layer
        model = Wrapped_Network(base, pretrained_model_args["num_classes"], 1).to(
            device
        )
    else:
        print("No pretrained model; creating a new Network instance.")
        # Create new model from scratch
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
        # Wrap the model with graph agg and a linear layer
        model = Wrapped_Network(base, model_args["num_classes"], 1).to(device)

    # Optimizing
    optimizer = torch.optim.Adam(model.parameters(), lr=model_args["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        verbose=True,
        min_lr=model_args["lr"] / 1e3,
    )

    # Early stopping parameters
    patience = 50
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Define loss
    criterion = torch.nn.MSELoss()

    # -- Training loop with logging --
    with open(training_log_path, "w") as log_file:
        log_file.write("epoch,train_loss,val_loss,test_loss\n")

        for epoch in range(1, model_args["epochs"] + 1):
            # Train for one epoch
            loss = train_mse(model, train_loader, optimizer, criterion, device)

            # Compute stats on train / val / test
            train_loss = test_mse(model, train_loader, criterion, device)
            val_loss = test_mse(model, val_loader, criterion, device)
            test_loss = test_mse(model, test_loader, criterion, device)

            # Print to console
            print(f"\nEpoch: {epoch:03d}, During Training Loss: {loss:.4f}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}")

            # Write to the training log file
            log_file.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{test_loss:.4f}\n")
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
    best_model_args = model_args if pretrained_dir is None else pretrained_model_args
    base = Network(
        irreps_in=o3.Irreps(best_model_args["irreps_in"]),
        irreps_hidden=o3.Irreps(best_model_args["irreps_hidden"]),
        irreps_out=o3.Irreps(best_model_args["irreps_out"]),
        irreps_node_attr=o3.Irreps(best_model_args["irreps_node_attr"]),
        irreps_edge_attr=o3.Irreps(best_model_args["irreps_edge_attr"]),
        layers=best_model_args["layers"],
        max_radius=best_model_args["max_radius"],
        number_of_basis=best_model_args["number_of_basis"],
        radial_layers=best_model_args["radial_layers"],
        radial_neurons=best_model_args["radial_neurons"],
        num_neighbors=best_model_args["num_neighbors"],
        num_nodes=best_model_args["num_nodes"],
        reduce_output=best_model_args["reduce_output"],
        num_classes=best_model_args["num_classes"],
        num_heads=best_model_args["num_heads"],
    ).to(device)
    best_model = Wrapped_Network(base, best_model_args["num_classes"], 1).to(device)
    best_model.load_state_dict(torch.load(best_model_path))

    # -- Final stats with the best model --
    final_train_loss = test_mse(best_model, train_loader, criterion, device)
    final_val_loss = test_mse(best_model, val_loader, criterion, device)
    final_test_loss = test_mse(best_model, test_loader, criterion, device)

    # Print
    print(f"\n===== Final Stats (Best Model) =====")
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


def train_mse(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        data = process_batch(data)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(-1), data.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return total_loss / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def test_mse(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for data in tqdm(loader, desc="Testing", leave=False):
        data = data.to(device)
        data = process_batch(data)
        outputs = model(data)
        loss = criterion(outputs.squeeze(-1), data.y.float())

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return total_loss / total_samples if total_samples > 0 else 0.0


class Wrapped_Network(torch.nn.Module):
    def __init__(self, base, input_dim, output_dim):
        super().__init__()
        self.base = base
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = torch.nn.Linear(5 * input_dim, output_dim)

    def forward(self, data):
        # Base network
        x = self.base(data).squeeze()  # General squeeze for redundant dimensions

        # sum, mean, min, max
        x_sum = scatter(x, data.batch, dim=0, reduce="sum")
        x_mean = scatter(x, data.batch, dim=0, reduce="mean")
        x_min = scatter(x, data.batch, dim=0, reduce="min")
        x_max = scatter(x, data.batch, dim=0, reduce="max")

        # manual std
        mean_val_sq = scatter(x ** 2, data.batch, dim=0, reduce="mean")
        std_val = (mean_val_sq - x_mean ** 2).clamp_min(1e-5).sqrt()
        x = torch.cat([x_sum, x_mean, x_min, x_max, std_val], dim=1)

        return self.embedding(x)


def main():
    pretrained_dir = "./logs/composite_2025-01-13_23-24-31/"
    # dataset_sizes = [100, 1000]
    # for dataset_size in dataset_sizes:
    #     # train_model(None, dataset_size)
    #     train_model(pretrained_dir, dataset_size)
    train_model(None, 10000)
    train_model(pretrained_dir, 10000)


if __name__ == "__main__":
    main()
