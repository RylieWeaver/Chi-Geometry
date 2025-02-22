# General
from tqdm import tqdm

# Torch
import torch

# Custom
from examples.utils import process_batch


# Training Function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        data = process_batch(data)

        # Compute loss
        pred = model(data).squeeze()  # Shape: [num_nodes] or [num_nodes, output_dim]
        true = data.y.squeeze()  # Shape: [num_nodes] or [num_nodes, output_dim]
        loss = criterion(pred, true)
        total_loss += (
            loss.item() * data.num_graphs
        )  # De-normalize by batch size for reporting

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

    return total_loss / len(loader.dataset)


# Testing Functions
def test_classification(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    correct_per_class = torch.zeros(num_classes, device=device)
    total_per_class = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for data in tqdm(loader, desc="Testing", leave=False):
            data = data.to(device)
            data = process_batch(data)

            # Compute loss and accuracy
            pred = model(data)  # Shape: [num_nodes, num_classes]
            true = data.y.squeeze()  # Shape: [num_nodes]
            loss = criterion(pred, true)
            total_loss += (
                loss.item() * data.num_graphs
            )  # De-normalize by batch size for reporting
            predictions = pred.argmax(dim=1)  # Shape: [num_nodes]

            # Compute per-class correct and total counts
            for cls in range(num_classes):
                cls_mask = true == cls
                correct = (predictions[cls_mask] == cls).sum().item()
                total = cls_mask.sum().item()
                correct_per_class[cls] += correct
                total_per_class[cls] += total

    # Compute per-class accuracies by averaging over all heads
    per_class_accuracies = correct_per_class / (total_per_class + 1e-10)

    return total_loss / len(loader.dataset), per_class_accuracies.cpu().numpy()


def test_regression(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="Testing", leave=False):
            data = data.to(device)
            data = process_batch(data)

            # Compute loss
            pred = model(data)  # Shape: [num_nodes, output_dim]
            true = data.y.view(-1, 1)  # Shape: [num_nodes, output_dim]
            loss = criterion(pred, true)
            total_loss += (
                loss.item() * data.num_graphs
            )  # De-normalize by batch size for reporting

    return total_loss / len(loader.dataset)


def shuffle_split_dataset(dataset, seed=42):
    # Dataset
    torch.manual_seed(seed)
    shuffled_indices = torch.randperm(len(dataset))
    shuffled_dataset = [dataset[i] for i in shuffled_indices]

    train_size = int(0.8 * len(shuffled_dataset))
    val_size = int(0.1 * len(shuffled_dataset))
    train_dataset = shuffled_dataset[:train_size]
    val_dataset = shuffled_dataset[train_size : train_size + val_size]
    test_dataset = shuffled_dataset[train_size + val_size :]

    return train_dataset, val_dataset, test_dataset
