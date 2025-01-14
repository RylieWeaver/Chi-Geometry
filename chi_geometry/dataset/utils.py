# General
import json

# Torch
import torch

# Etc
from scipy.spatial.transform import Rotation


def load_dataset_json(file_path):
    # Default
    default_args = {
        "num_samples": 3000,
        "type": "simple",
        "chirality_distance": 1,
        "species_range": 8,
        "points": 4,
        "save_path": "dataset.pt",
        "noise": False,
    }

    # Read
    try:
        with open(file_path, "r") as f:
            config_args = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found at {file_path}. Using default arguments.")
        config_args = {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}. Using default arguments.")
        config_args = {}

    # Update
    args = {**default_args, **config_args}

    return args


def center_and_rotate_positions(pos):
    """
    Centers the positions at zero and applies a random rotation.

    Parameters:
    data (torch_geometric.data.Data): The data object containing node positions in data.pos

    Returns:
    torch_geometric.data.Data: The data object with rotated positions
    """
    # Ensure positions are in the correct shape [num_nodes, 3]
    assert (
        pos.dim() == 2 and pos.size(1) == 3
    ), "data.pos must be of shape [num_nodes, 3]"

    # Step 1: Center the positions at zero
    pos_centered = pos - pos.mean(dim=0, keepdim=True)

    # Step 2: Generate a uniformly random rotation matrix
    # Using scipy's Rotation module to generate a random rotation
    r = (
        Rotation.random()
    )  # Generates a random rotation (uniformly distributed over SO(3))
    rotation_matrix = torch.tensor(
        r.as_matrix(), dtype=pos_centered.dtype, device=pos_centered.device
    )  # Shape [3, 3]

    # Step 3: Apply the rotation to the centered positions
    pos_rotated = torch.matmul(pos_centered, rotation_matrix.T)  # Shape [num_nodes, 3]

    # Update the data object with the rotated positions
    pos = pos_rotated

    return pos
