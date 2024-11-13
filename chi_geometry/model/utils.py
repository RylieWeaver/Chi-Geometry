# General
import json

# Torch
from torch_scatter import scatter


def load_model_json(file_path):
    # Default arguments
    default_args = {
        # Learning parameters
        'epochs': 75,
        'batch_size': 128,
        'lr': 0.001,
        'use_cuda': False,
        # Model parameters
        'input_dim': "40",
        'hidden_dim': "20",
        'output_dim': "10",
        'irreps_node_attr': "1x0e",
        'irreps_edge_attr': "1x0e + 1x1o",
        'layers': 4,
        'max_radius': 2.0,
        'number_of_basis': 5,
        'radial_layers': 2,
        'radial_neurons': 50,
        'num_neighbors': 1.5,
        'num_nodes': 4.0,
        'reduce_output': False,
        'num_classes': 3,
        "class_weights": [1.0, 10.0, 10.0],
        'num_heads': 1
    }

    # Read configuration from JSON file
    try:
        with open(file_path, 'r') as f:
            config_args = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found at {file_path}. Using default arguments.")
        config_args = {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}. Using default arguments.")
        config_args = {}

    # Merge default arguments with those from the config file
    args = {**default_args, **config_args}

    # Make irreps from dims
    args["irreps_in"] = f"{args['input_dim']}x0e"
    args["irreps_hidden"] = create_irreps_string(int(args["hidden_dim"]), 1)
    args["irreps_out"] = create_irreps_string(int(args["output_dim"]), 0)
    
    return args


def process_batch(data):
    # Center the positions (necessary for E(3) equivariance models)
    mean_pos = scatter(data.pos, data.batch, dim=0, reduce="mean")
    data.pos = data.pos - mean_pos[data.batch]

    return data


def create_irreps_string(node_dim: int, max_ell: int):
    """Create a string representation of the irreps for the E(3) layer"""
    irreps = []
    for ell in range(max_ell + 1):
        irreps.append(f"{node_dim}x{ell}e")
        irreps.append(f"{node_dim}x{ell}o")
    return " + ".join(irreps)


