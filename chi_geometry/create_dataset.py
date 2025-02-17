# General
import os
import warnings
import random
import math
import numpy as np
from itertools import combinations

# Torch
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# Chi-Geometry
from chi_geometry import load_dataset_json, center_and_rotate_positions


def scalar_triple_product(v1, v2, v3):
    return np.dot(v1, np.cross(v2, v3))


# NOTE Classic chiral configuration as defined in chemistry
def create_classic_chiral_instance(chirality_distance=1, species_range=10, noise=False):
    # Step 0: Assert necessary conditions
    assert (
        chirality_distance >= 1
    ), "Distance must be greater than 0. We can't have a chiral center without any connections"
    assert (
        species_range >= chirality_distance + 4
    ), "Range must be greater than distance+4 to provide enough unique species for the chiral configuration"

    # Step 1: Choose random atomic numbers without replacement
    atomic_numbers = random.sample(
        range(1, species_range + 1), 5 + (chirality_distance - 1)
    )

    # Step 2: Choose the atom type for the chiral center
    chiral_center = random.choice(atomic_numbers)

    # Step 3: Choose the atom types for the intermediate layers
    if chirality_distance > 1:
        # Exclude the chiral center when selecting intermediate layer atoms
        available_atoms = [atom for atom in atomic_numbers if atom != chiral_center]
        intermediate_layers = random.sample(available_atoms, chirality_distance - 1)
    else:
        intermediate_layers = []

    # Step 4: Select the Chiral substituent quadruplet
    # Exclude both the chiral center and intermediate layer atoms
    quadruplet_atoms = [
        atom
        for atom in atomic_numbers
        if atom != chiral_center and atom not in intermediate_layers
    ]
    lowest_priority_atom = [min(quadruplet_atoms)]
    triplet_atoms = [
        atom for atom in quadruplet_atoms if atom not in lowest_priority_atom
    ]

    # ------------------------------------
    # Step 5: Assign positions with layers
    # ------------------------------------
    base_angles = [
        0,
        2 * math.pi / 3,
        4 * math.pi / 3,
    ]  # Clockwise when viewed from negative z-axis
    positions = []
    z_layer = 1.0
    z_toplayer = 1.0
    z_bottomlayer = 1.0

    # Step 5.1: Chiral Center position
    if not noise:
        positions.append([0.0, 0.0, z_layer])
    else:
        center_radius = random.uniform(0.0, 1.0)
        center_angle = random.uniform(0.0, 2 * math.pi)
        positions.append(
            [
                math.cos(center_angle) * center_radius,
                math.sin(center_angle) * center_radius,
                z_layer,
            ]
        )

    # Step 5.2: Following layer positions
    for _ in range(chirality_distance):
        if not noise:
            layer_distance = 0.5  # deterministic
            top_angle_noise = 0
            bottom_angle_noises = [0, 0, 0]
            top_radius = 0.0
            bottom_radius = 1.0
        else:
            layer_distance = random.uniform(0.1, 1.5)
            top_angle_noise = random.uniform(-math.pi, math.pi)
            bottom_angle_noises = [
                random.uniform(-math.pi / 3.1, math.pi / 3.1) for _ in range(3)
            ]
            top_radius = random.uniform(0.0, 1.0)
            bottom_radius = random.uniform(0.1, 1.0)

        # Move down the z-axis for this layer
        z_toplayer += layer_distance
        positions.append(
            [
                math.cos(top_angle_noise) * top_radius,
                math.sin(top_angle_noise) * top_radius,
                z_toplayer,
            ]
        )
        z_bottomlayer -= layer_distance
        # Append 3 new points for the bottom part of this layer
        for angle, angle_noise in zip(base_angles, bottom_angle_noises):
            final_angle = angle + angle_noise
            x = math.cos(final_angle) * bottom_radius
            y = math.sin(final_angle) * bottom_radius
            z = z_bottomlayer
            positions.append([x, y, z])
    positions = torch.tensor(positions, dtype=torch.float)

    # Step 6: Create edge connections
    edges = []
    # Connect chiral center to lowest priority and substituent atoms (atom indices 1, 2, 3, 4)
    for i in range(1, 5):
        edges.append([0, i])
        edges.append([i, 0])
    # Connect intermediate atoms to the next atoms in the substituent chains
    for l in range(1, chirality_distance):
        for i in range(4):
            src = 4 * (l - 1) + i + 1
            dst = 4 * l + i + 1
            edges.append([src, dst])
            edges.append([dst, src])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Step 7: Sort triplet atoms in ascending or descending order. This should correspond
    #         to the chirality just because of how we have set up appending the positions.
    ascending = random.choice([True, False])
    triplet_atoms.sort(reverse=ascending)
    quadruplet_priority_idx = (
        [1, 4, 3, 2] if ascending else [1, 2, 3, 4]
    )  # idx of atoms from lowest to highest priority

    # Step 8: Assign scalar triple product and check with expected
    center_pos = positions[0]
    stp = scalar_triple_product(
        positions[quadruplet_priority_idx[0]] - center_pos,
        positions[quadruplet_priority_idx[2]] - positions[quadruplet_priority_idx[3]],
        positions[quadruplet_priority_idx[1]] - positions[quadruplet_priority_idx[2]],
    )
    if stp > 0:
        chirality_value = 1
        chirality_tag = [0, 1, 0]
        chirality_str = "R"
    elif stp < 0:
        chirality_value = 2
        chirality_tag = [0, 0, 1]
        chirality_str = "S"
    else:
        raise ValueError("Scalar triple product is 0")

    # Step 9: Assign chirality tags
    chirality = [chirality_value] + [0] * (4 * chirality_distance)
    chirality_tags = [chirality_tag] + [[1, 0, 0]] * (4 * chirality_distance)
    chirality_strs = [chirality_str] + ["N/A"] * (4 * chirality_distance)
    chirality = torch.tensor(chirality, dtype=torch.float).unsqueeze(-1)
    chirality_one_hot = torch.tensor(chirality_tags, dtype=torch.float)

    # Step 10: Create node features and make sure they're tensors
    atomic_numbers = torch.tensor(
        [chiral_center]
        + [atom for atom in intermediate_layers for _ in range(4)]
        + lowest_priority_atom
        + triplet_atoms,
        dtype=torch.int64,
    ).view(-1, 1)
    atomic_numbers_one_hot = (
        F.one_hot(atomic_numbers.squeeze() - 1, num_classes=118)
    ).float()  # Subtract 1 because atomic numbers start from 1

    # Step 11: Apply centering and random rotation to positions
    # positions = center_and_rotate_positions(positions)

    # Step 12: Construct PyTorch Geometric data object
    data = Data(
        x=atomic_numbers_one_hot.float(),
        y=chirality,
        z=atomic_numbers.float(),
        atomic_numbers=atomic_numbers,
        atomic_numbers_one_hot=atomic_numbers_one_hot,
        chirality=chirality,
        chirality_one_hot=chirality_one_hot,
        chirality_str=chirality_strs,
        edge_index=edge_index,
        pos=positions,
    )

    return data


# NOTE Simple chiral configurations have simple connections between layers
def create_simple_chiral_instance(chirality_distance=1, species_range=10, noise=False):
    # Step 0: Assert necessary conditions
    assert (
        chirality_distance >= 1
    ), "Distance must be greater than 0. We can't have a chiral center without any connections"
    assert (
        species_range >= chirality_distance + 3
    ), "Range must be greater than distance+3 to provide enough unique species for the chiral configuration"

    # Step 1: Choose random atomic numbers without replacement
    atomic_numbers = random.sample(
        range(1, species_range + 1), 4 + (chirality_distance - 1)
    )

    # Step 2: Choose the atom type for the chiral center
    chiral_center = random.choice(atomic_numbers)

    # Step 3: Choose the atom types for the intermediate layers
    if chirality_distance > 1:
        # Exclude the chiral center when selecting intermediate layer atoms
        available_atoms = [atom for atom in atomic_numbers if atom != chiral_center]
        intermediate_layers = random.sample(available_atoms, chirality_distance - 1)
    else:
        intermediate_layers = []

    # Step 4: Select the Chiral substituent triplet
    # Exclude both the chiral center and intermediate layer atoms
    triplet_atoms = [
        atom
        for atom in atomic_numbers
        if atom != chiral_center and atom not in intermediate_layers
    ]

    # ------------------------------------
    # Step 5: Assign positions with layers
    # ------------------------------------
    base_angles = [
        0,
        2 * math.pi / 3,
        4 * math.pi / 3,
    ]  # Clockwise when viewed from negative z-axis
    positions = []
    z_layer = 1.0

    # Step 5.1: Chiral Center position
    if not noise:
        positions.append([0.0, 0.0, z_layer])
    else:
        center_radius = random.uniform(0.0, 1.0)
        center_angle = random.uniform(0.0, 2 * math.pi)
        positions.append(
            [
                math.cos(center_angle) * center_radius,
                math.sin(center_angle) * center_radius,
                z_layer,
            ]
        )

    # Step 5.2: Following layer positions
    for _ in range(chirality_distance):
        if not noise:
            layer_distance = 0.5  # deterministic
            angle_noises = [0, 0, 0]
            radius = 1.0
        else:
            layer_distance = random.uniform(0.1, 1.5)
            angle_noises = [
                random.uniform(-math.pi / 3.1, math.pi / 3.1) for _ in range(3)
            ]
            radius = random.uniform(0.1, 1.0)

        # Move down the z-axis for this layer
        z_layer -= layer_distance
        # Append 3 new points for this layer
        for angle, angle_noise in zip(base_angles, angle_noises):
            final_angle = angle + angle_noise
            x = math.cos(final_angle) * radius
            y = math.sin(final_angle) * radius
            z = z_layer
            positions.append([x, y, z])
    positions = torch.tensor(positions, dtype=torch.float)

    # Step 6: Create edge connections
    edges = []
    # Connect chiral center (0) to first layer atoms (1, 2, 3)
    for i in range(1, 4):
        edges.append([0, i])
        edges.append([i, 0])
    # Connect each atom in intermediate layers to the atom directly beneath
    for layer in range(1, chirality_distance):
        for i in range(3):
            src = 3 * (layer - 1) + 1 + i
            dst = 3 * layer + 1 + i
            edges.append([src, dst])
            edges.append([dst, src])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Step 7: Sort triplet atoms in ascending or descending order. This should correspond
    #         to the chirality just because of how we have set up appending the positions.
    ascending = random.choice(
        [True, False]
    )  # ascending has the largest atomic number first
    triplet_atoms.sort(reverse=ascending)
    triplet_priority_idx = (
        [3, 2, 1] if ascending else [1, 2, 3]
    )  # idx of atoms from lowest to highest priority

    # Step 8: Assign scalar triple product and check with expected
    stp = scalar_triple_product(
        positions[0]
        - positions[
            triplet_priority_idx[2]
        ],  # Relative vector from highest priority neighbor to the chiral center
        positions[0]
        - positions[
            triplet_priority_idx[1]
        ],  # Relative vector from second highest priority neighbor to the chiral center
        positions[0]
        - positions[
            triplet_priority_idx[0]
        ],  # Relative vector from lowest priority neighbor to the chiral center
    )
    if stp > 0:
        chirality_value = 1
        chirality_tag = [0, 1, 0]
        chirality_str = "R"
    elif stp < 0:
        chirality_value = 2
        chirality_tag = [0, 0, 1]
        chirality_str = "S"
    else:
        raise ValueError("Scalar triple product is 0")

    # Step 9: Assign chirality tags
    chirality = [chirality_value] + [0] * (3 * chirality_distance)
    chirality_tags = [chirality_tag] + [[1, 0, 0]] * (3 * chirality_distance)
    chirality_strs = [[chirality_str] + ["N/A"] * (3 * chirality_distance)]
    chirality = torch.tensor(chirality, dtype=torch.float).unsqueeze(-1)
    chirality_one_hot = torch.tensor(chirality_tags, dtype=torch.float)

    # Step 10: Create node features and make sure they're tensors
    atomic_numbers = torch.tensor(
        [chiral_center]
        + [atom for atom in intermediate_layers for _ in range(3)]
        + triplet_atoms,
        dtype=torch.int64,
    ).view(-1, 1)
    atomic_numbers_one_hot = (
        F.one_hot(atomic_numbers.squeeze() - 1, num_classes=118)
    ).float()  # Subtract 1 because atomic numbers start from 1

    # Step 11: Apply centering and random rotation to positions
    positions = center_and_rotate_positions(positions)

    # Step 12: Construct PyTorch Geometric data object
    data = Data(
        x=atomic_numbers_one_hot.float(),
        y=chirality,
        z=atomic_numbers.float(),
        atomic_numbers=atomic_numbers,
        atomic_numbers_one_hot=atomic_numbers_one_hot,
        chirality=chirality,
        chirality_one_hot=chirality_one_hot,
        chirality_str=chirality_strs,
        edge_index=edge_index,
        pos=positions,
    )

    return data


# NOTE Crossed chiral configurations have crossed connections between layers
def create_crossed_chiral_instance(chirality_distance=1, species_range=10, noise=False):
    # Step 0: Assert necessary conditions
    assert (
        chirality_distance >= 1
    ), "Distance must be greater than 0. We can't have a chiral center without any connections"
    assert (
        species_range >= chirality_distance + 3
    ), "Range must be greater than distance+3 to provide enough unique species for the chiral configuration"

    # Step 1: Choose 4 random atomic numbers without replacement
    atomic_numbers = random.sample(
        range(1, species_range), 4 + (chirality_distance - 1)
    )  # 119 is the highest atomic number

    # Step 2: Choose the atom type for the chiral center
    chiral_center = random.choice(atomic_numbers)

    # Step 3: Choose the atom types for the intermediate layers
    if chirality_distance > 1:
        # Exclude the chiral center when selecting intermediate layer atoms
        available_atoms = [atom for atom in atomic_numbers if atom != chiral_center]
        intermediate_layers = random.sample(available_atoms, chirality_distance - 1)
    else:
        intermediate_layers = []

    # Step 4: Select the Chiral substituent triplet
    # Exclude both the chiral center and intermediate layer atoms
    triplet_atoms = [
        atom
        for atom in atomic_numbers
        if atom != chiral_center and atom not in intermediate_layers
    ]

    # ------------------------------------
    # Step 5: Assign positions with layers
    # ------------------------------------
    base_angles = [
        0,
        2 * math.pi / 3,
        4 * math.pi / 3,
    ]  # Clockwise when viewed from negative z-axis
    positions = []
    z_layer = 1.0

    # Step 5.1: Chiral Center position
    if not noise:
        positions.append([0.0, 0.0, z_layer])
    else:
        center_radius = random.uniform(0.0, 1.0)
        center_angle = random.uniform(0.0, 2 * math.pi)
        positions.append(
            [
                math.cos(center_angle) * center_radius,
                math.sin(center_angle) * center_radius,
                z_layer,
            ]
        )

    # Step 5.2: Following layer positions
    for _ in range(chirality_distance):
        if not noise:
            layer_distance = 0.5  # deterministic
            angle_noises = [0, 0, 0]
            radius = 1.0
        else:
            layer_distance = random.uniform(0.1, 1.5)
            angle_noises = [
                random.uniform(-math.pi / 3.1, math.pi / 3.1) for _ in range(3)
            ]
            radius = random.uniform(0.1, 1.0)

        # Move down the z-axis for this layer
        z_layer -= layer_distance
        # Append 3 new points for this layer
        for angle, angle_noise in zip(base_angles, angle_noises):
            final_angle = angle + angle_noise
            x = math.cos(final_angle) * radius
            y = math.sin(final_angle) * radius
            z = z_layer
            positions.append([x, y, z])
    positions = torch.tensor(positions, dtype=torch.float)

    # Step 6: Create edge connections
    edges = []
    # Connect chiral center (0) to first layer atoms (1, 2, 3)
    for i in range(1, 4):
        edges.append([0, i])
        edges.append([i, 0])
    # Connect each atom in intermediate layers to a random atom in the layer directly beneath in a 1-to-1 fashion
    for layer in range(1, chirality_distance):
        # Generate a random permutation of [0, 1, 2] for 1-to-1 connections
        permutation = random.sample([0, 1, 2], 3)
        for i in range(3):
            src = 3 * (layer - 1) + 1 + i
            dst = 3 * layer + 1 + permutation[i]
            edges.append([src, dst])
            edges.append([dst, src])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Step 7: Sort triplet atoms in ascending or descending order. This should correspond
    #         to the chirality just because of how we have set up appending the positions.
    ascending = random.choice(
        [True, False]
    )  # ascending has the largest atomic number first
    triplet_atoms.sort(reverse=ascending)
    triplet_priority_idx = (
        [-1, -2, -3] if ascending else [-3, -2, -1]
    )  # idx of atoms from lowest to highest priority

    # Step 8: Assign scalar triple product
    stp = scalar_triple_product(
        positions[0]
        - positions[
            triplet_priority_idx[2]
        ],  # Relative vector from highest priority neighbor to the chiral center
        positions[0]
        - positions[
            triplet_priority_idx[1]
        ],  # Relative vector from second highest priority neighbor to the chiral center
        positions[0]
        - positions[
            triplet_priority_idx[0]
        ],  # Relative vector from lowest priority neighbor to the chiral center
    )
    if stp > 0:
        chirality_value = 1
        chirality_tag = [0, 1, 0]
        chirality_str = "R"
    elif stp < 0:
        chirality_value = 2
        chirality_tag = [0, 0, 1]
        chirality_str = "S"
    else:
        raise ValueError("Scalar triple product is 0")

    # Step 9: Assign chirality tags
    chirality = [chirality_value] + [0] * (3 * chirality_distance)
    chirality_tags = [chirality_tag] + [[1, 0, 0]] * (3 * chirality_distance)
    chirality_strs = [chirality_str] + ["N/A"] * (3 * chirality_distance)
    chirality = torch.tensor(chirality, dtype=torch.float).unsqueeze(-1)
    chirality_one_hot = torch.tensor(chirality_tags, dtype=torch.float)

    # Step 10: Create node features and make sure they're tensors
    atomic_numbers = torch.tensor(
        [chiral_center]
        + [atom for atom in intermediate_layers for _ in range(3)]
        + triplet_atoms,
        dtype=torch.int64,
    ).view(-1, 1)
    atomic_numbers_one_hot = (
        F.one_hot(atomic_numbers.squeeze() - 1, num_classes=118)
    ).float()  # Subtract 1 because atomic numbers start from 1

    # Step 11: Apply centering and random rotation to positions
    positions = center_and_rotate_positions(positions)

    # Step 12: Construct PyTorch Geometric data object
    data = Data(
        x=atomic_numbers_one_hot.float(),
        y=chirality,
        z=atomic_numbers.float(),
        atomic_numbers=atomic_numbers,
        atomic_numbers_one_hot=atomic_numbers_one_hot,
        chirality=chirality,
        chirality_one_hot=chirality_one_hot,
        chirality_str=chirality_strs,
        edge_index=edge_index,
        pos=positions,
    )

    return data


def create_chiral_instance(
    chirality_type="simple",
    chirality_distance=1,
    species_range=15,
    noise=False,
):
    assert (
        species_range <= 118
    ), "Species range must be less than or equal to 118 (number of elements in the periodic table)"

    if chirality_type == "classic":
        return create_classic_chiral_instance(
            chirality_distance=chirality_distance,
            species_range=species_range,
            noise=noise,
        )
    elif chirality_type == "simple":
        return create_simple_chiral_instance(
            chirality_distance=chirality_distance,
            species_range=species_range,
            noise=noise,
        )
    elif chirality_type == "crossed":
        return create_crossed_chiral_instance(
            chirality_distance=chirality_distance,
            species_range=species_range,
            noise=noise,
        )
    else:
        raise ValueError(f"Chiral type not supported: {chirality_type}")


def create_dataset(
    num_samples=3000,
    chirality_type="simple",
    chirality_distance=1,
    species_range=10,
    noise=False,
):
    # Communication
    print("Creating dataset with the following parameters:")
    print(f"Number of samples: {num_samples}")
    print(f"Chirality Type: {chirality_type}")
    print(f"Chirality distance: {chirality_distance}")
    print(f"Species range: {species_range}")
    print(f"Noise: {noise}")

    # Create
    data_list = []
    for _ in range(num_samples):
        # Generate chiral graph
        data = create_chiral_instance(
            chirality_type, chirality_distance, species_range, noise
        )
        # Append
        data_list.append(data)

    return data_list


def main():
    # Read Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    args = load_dataset_json(config_path)

    # Extract arguments
    num_samples = args["num_samples"]
    chirality_type = args["chirality_type"]
    chirality_distance = args["chirality_distance"]
    species_range = args["species_range"]
    save_path = args["save_path"]
    noise = args["noise"]

    # Create
    print("Creating dataset...")
    dataset = create_dataset(
        num_samples=num_samples,
        chirality_type=chirality_type,
        chirality_distance=chirality_distance,
        species_range=species_range,
        noise=noise,
    )

    # Save
    if os.path.dirname(save_path):  # Check if there is a directory in the path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, save_path)
    print(f"Dataset saved as {save_path}")
    print("Dataset created.")


if __name__ == "__main__":
    main()
