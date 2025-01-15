# General
from tqdm import tqdm

# Torch
import torch

# Custom
from chi_geometry.model import process_batch


# Training Function
def train(model, loader, optimizer, criterion, device):
    model.train()
    num_heads = len(model.OutModule)
    total_loss = 0
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        data = process_batch(data)
        outputs = model(data)  # Shape: [num_heads, num_nodes, num_classes]

        # Initialize loss
        loss = 0
        for head_idx in range(num_heads):
            # Get outputs and targets for this head
            head_output = outputs[head_idx]  # Shape: [num_nodes, num_classes]
            head_target = data.y[:, head_idx].long()  # Shape: [num_nodes]

            # Compute loss for this head
            head_loss = criterion(head_output, head_target)
            loss += head_loss / num_heads  # Sum losses over heads

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


# Testing Function
def test(model, loader, criterion, device, num_classes):
    model.eval()
    num_heads = len(outputs.size(0))
    correct_per_class = torch.zeros(num_classes, device=device)
    total_per_class = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for data in tqdm(loader, desc="Testing", leave=False):
            data = data.to(device)
            data = process_batch(data)
            outputs = model(data)  # Shape: [num_heads, num_nodes, num_classes]

            # Initialize loss
            loss = 0
            for head_idx in range(num_heads):
                # Get outputs and targets for this head
                head_output = outputs[head_idx]  # Shape: [num_nodes, num_classes]
                head_target = data.y[:, head_idx].long()  # Shape: [num_nodes]

                # Compute loss for this head
                head_loss = criterion(head_output, head_target)
                loss += head_loss / num_heads  # Average over heads

                # Get predictions for this head
                predictions = head_output.argmax(dim=1)  # Shape: [num_nodes]

                # Compute per-class correct and total counts
                for cls in range(num_classes):
                    cls_mask = head_target == cls
                    correct = (predictions[cls_mask] == cls).sum().item()
                    total = cls_mask.sum().item()
                    correct_per_class[cls] += correct
                    total_per_class[cls] += total

            total_loss += loss.item() * data.num_graphs

    # Compute per-class accuracies by averaging over all heads
    per_class_accuracies = correct_per_class / (total_per_class + 1e-10)

    return total_loss / len(loader.dataset), per_class_accuracies.cpu().numpy()
