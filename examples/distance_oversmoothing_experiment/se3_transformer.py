# General
import os
import json
import datetime

# Torch
import torch
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Equivariant Neural Networks
from e3nn import o3
from equiformer_pytorch import Equiformer

# Custom
from chi_geometry.dataset import load_dataset_json, create_dataset
from chi_geometry.model import load_model_json, train, test
from utils import make_global_connections


class WrappedEquiformer(torch.nn.Module):
    """
    A simple Equiformer-based model for classification.
    Uses three degrees (l=0,1,2) with specified hidden dimensions,
    then maps the degree-0 output to 'num_classes'.

    There is reshaping and masking logic because transformers expect
    a batch dimension with constant graph size.
    """

    def __init__(self, input_dim, hidden_dim, layers, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.num_classes = num_classes

        self.input_embedding = torch.nn.Linear(
            118, self.hidden_dim
        )  # Embed one-hot atom type
        # Equiformer setup:
        #  - degree-0 dimension = hidden_dim
        #  - degree-1 dimension = hidden_dim//2
        #  - degree-2 dimension = hidden_dim//8
        self.equiformer = Equiformer(
            dim=(self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 8),
            num_degrees=3,  # l=0,1,2
            depth=self.layers,
            num_linear_attn_heads=1,
            reduce_dim_out=False,
            attend_self=True,
            heads=(4, 1, 1),
            dim_head=(self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 8),
        )
        self.output_embedding = torch.nn.Linear(
            self.hidden_dim, self.num_classes
        )  # Embed down for classification

    def reshape_batch(
        self, feats: torch.Tensor, coords: torch.Tensor, batch: torch.Tensor
    ):
        """
        feats:  [total_nodes, feature_dim]  (concatenated from multiple graphs)
        coords: [total_nodes, 3]           (xyz positions for each node)
        batch:  [total_nodes]              (which graph each node belongs to, in range [0..B-1])

        Returns:
        feats_3d:  [B, N_max, feature_dim]
        coords_3d: [B, N_max, 3]
        mask:      [B, N_max] (True where node is real, False where padded)
        """
        device = feats.device

        # 1) Figure out how many graphs we have
        num_graphs = batch.max().item() + 1

        # 2) Split feats and coords by graph
        feats_per_graph = []
        coords_per_graph = []
        lengths = []
        for g in range(num_graphs):
            node_indices = (batch == g).nonzero(as_tuple=True)[0]
            feats_per_graph.append(feats[node_indices])  # shape [n_g, feature_dim]
            coords_per_graph.append(coords[node_indices])  # shape [n_g, 3]
            lengths.append(len(node_indices))  # store how many nodes in this graph

        # 3) Pad each list to get a uniform node dimension = N_max
        #    feats_3d => [B, N_max, feature_dim]
        #    coords_3d => [B, N_max, 3]
        feats_3d = rnn_utils.pad_sequence(
            feats_per_graph, batch_first=True, padding_value=0.0
        )
        coords_3d = rnn_utils.pad_sequence(
            coords_per_graph, batch_first=True, padding_value=0.0
        )

        # 4) Build a boolean mask of shape [B, N_max], marking real vs. padded
        N_max = feats_3d.size(1)
        mask = torch.zeros((num_graphs, N_max), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        return feats_3d.to(device), coords_3d.to(device), mask

    def forward(self, data):
        # 0) Unpack data
        feats, coords, mask = self.reshape_batch(data.x, data.pos, data.batch)
        # 1) Embed down to hidden_dim
        feats = self.input_embedding(feats)  # [B, N, input_dim]
        # 2) Equiformer
        out = self.equiformer(
            feats,
            coords,
            mask=mask,  # shape [B, N]
        )
        # 3) Use degree-0 outputs for classification
        node_embs = out.type0  # [B, N, hidden_dim]
        logits = self.output_embedding(node_embs)  # [B, N, num_classes]
        # 4) Reshape and mask to expected node outputs
        logits = logits.view(-1, self.num_classes)  # [B*N, num_classes]
        logits = logits[mask.view(-1)].unsqueeze(
            0
        )  # [1, total_valid_nodes, num_classes]

        return logits


def main():
    # Read Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    model_config_path = os.path.join(script_dir, "model_config.json")
    dataset_args = load_dataset_json(dataset_config_path)
    model_args = load_model_json(model_config_path)

    # Log Directory
    run_dir = os.path.join(
        script_dir,
        f"logs/se3-transformer-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, "best_model.pt")
    training_log_path = os.path.join(run_dir, "training_log.txt")
    final_results_path = os.path.join(run_dir, "final_results.txt")

    # Save model args
    model_args_path = os.path.join(run_dir, "model_config.json")
    with open(model_args_path, "w") as f:
        json.dump(model_args, f)

    # Device
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
    dataset = torch.load(dataset_args["save_path"])
    dataset = make_global_connections(dataset)
    print(f"Dataset contains {len(dataset)} graphs.")

    # Shuffle and split dataset
    torch.manual_seed(42)
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
    model = WrappedEquiformer(
        input_dim=model_args["input_dim"],
        hidden_dim=model_args["hidden_dim"],
        layers=model_args["layers"],
        num_classes=num_classes,
    ).to(device)

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
    class_weights = torch.tensor(model_args["class_weights"], device=device)
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
    best_model = WrappedEquiformer(
        input_dim=model_args["input_dim"],
        hidden_dim=model_args["hidden_dim"],
        layers=model_args["layers"],
        num_classes=num_classes,
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


if __name__ == "__main__":
    main()
