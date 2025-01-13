# General
import os
import datetime

# Torch
import torch
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
            num_degrees=3,  # l = 0,1,2
            depth=self.layers,
            num_linear_attn_heads=1,  # optional: global linear attention heads
            reduce_dim_out=False,  # keep dimension at the end
            attend_self=True,
            heads=(4, 1, 1),  # heads per degree
            dim_head=(self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 8),
        )
        self.output_embedding = torch.nn.Linear(
            self.hidden_dim, self.num_classes
        )  # Embed down for classification

    def forward(self, data, mask=None):
        # 0) Unpack data
        feats, coords = data.x, data.pos
        # 1) Embed features from 118 -> input_dim
        feats = self.input_embedding(feats)  # [B, N, input_dim]
        # 2) Pass through the Equiformer
        out = self.equiformer(feats, coords, mask=mask)
        # out.type0 => [B, N, hidden_dim] for the degree-0 channel
        node_embs = out.type0
        # 3) Classification head
        logits = self.output_embedding(node_embs)  # [B, N, num_classes]
        return logits


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
    dataset = torch.load(dataset_args["save_path"])
    dataset = make_global_connections(dataset)
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
    model = WrappedEquiformer(
        input_dim=model_args["input_dim"],
        hidden_dim=model_args["hidden_dim"],
        output_dim=model_args["output_dim"],
        layers=model_args["layers"],
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_args["lr"])
    # Define loss
    class_weights = torch.tensor(model_args["class_weights"], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(1, model_args["epochs"] + 1):
        loss = train(model, train_loader, optimizer, criterion, device)
        train_loss, train_accuracies = test(
            model, train_loader, criterion, device, num_classes
        )
        test_loss, test_accuracies = test(
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


if __name__ == "__main__":
    main()
