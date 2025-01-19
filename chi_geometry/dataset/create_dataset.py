# General
import os
import random
import math
import numpy as np
from itertools import combinations

# Torch
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# Chi-Geometry
from chi_geometry.dataset import load_dataset_json, center_and_rotate_positions


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

    # Step 5: Randomly decide R or S configuration
    clockwise = random.choice([True, False])

    # Step 6: Arrange atoms
    if clockwise:
        # Sort quadruplet atoms in ascending order (smallest to largest)
        triplet_atoms.sort()
        chirality_value = 1
        chirality_tag = [0, 1, 0]
        chirality_str = "R"
    else:
        # Sort quadruplet atoms in descending order (largest to smallest)
        triplet_atoms.sort(reverse=True)
        chirality_value = 2
        chirality_tag = [0, 0, 1]
        chirality_str = "S"

    # ------------------------------------
    # Step 7: Assign positions with layers
    # ------------------------------------
    base_angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
    positions = []
    z_layer = 1.0
    z_toplayer = 1.0
    z_bottomlayer = 1.0

    # Step 7.1: Chiral Center position
    positions.append([0.0, 0.0, z_layer])

    # Step 7.2: Following layer positions
    for _ in range(chirality_distance):
        if not noise:
            layer_distance = 0.5  # deterministic
            angle_noises = [0, 0, 0, 0]
            z_noises = [0, 0, 0, 0]
            radii = [0.0, 1.0, 1.0, 1.0]
        else:
            layer_distance = random.uniform(0.3, 2.0)
            angle_noises = [random.uniform(-math.pi / 4, math.pi / 4) for _ in range(3)]
            z_noises = [random.uniform(-0.1, 0.1) for _ in range(4)]
            radii = [random.uniform(0.1, 1.0) for _ in range(3)]

        # Append 1 new point for the top layer
        z_toplayer += layer_distance
        z = z_toplayer + z_noises[0]
        positions.append([0, 0, z])
        # Append 3 new points for the bottom layer
        z_bottomlayer -= layer_distance
        for angle, angle_noise, radius, z_noise in zip(
            base_angles, angle_noises, radii, z_noises[1:]
        ):
            final_angle = angle + angle_noise
            x = math.cos(final_angle) * radius
            y = math.sin(final_angle) * radius
            z = z_bottomlayer + z_noise
            positions.append([x, y, z])

    # Step 8: Create edge connections
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

    # Step 9: Assign chirality tags
    chirality = [chirality_value] + [0] * (4 * chirality_distance)
    chirality_tags = [chirality_tag] + [[1, 0, 0]] * (4 * chirality_distance)
    chirality_strs = [chirality_str] + ["N/A"] * (4 * chirality_distance)

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
    chirality = torch.tensor(chirality, dtype=torch.float).unsqueeze(-1)
    chirality_one_hot = torch.tensor(chirality_tags, dtype=torch.float)
    positions = torch.tensor(positions, dtype=torch.float)

    # Step 11: Create edge index and make sure it's a tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Step 12: Apply centering and random rotation to positions
    positions = center_and_rotate_positions(positions)

    # Step 13: Construct PyTorch Geometric data object
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

    # Step 5: Randomly decide clockwise or counterclockwise placement
    clockwise = random.choice([True, False])

    # Step 6: Arrange atoms
    if clockwise:
        # Sort other atoms in ascending order (smallest to largest)
        triplet_atoms.sort()
        chirality_value = 1
        chirality_tag = [0, 1, 0]
        chirality_str = "R"
    else:
        # Sort other atoms in descending order (largest to smallest)
        triplet_atoms.sort(reverse=True)
        chirality_value = 2
        chirality_tag = [0, 0, 1]
        chirality_str = "S"

    # ------------------------------------
    # Step 7: Assign positions with layers
    # ------------------------------------
    base_angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
    positions = []
    z_layer = 1.0

    # Step 7.1: Chiral Center position
    positions.append([0.0, 0.0, z_layer])

    # Step 7.2: Following layer positions
    for _ in range(chirality_distance):
        if not noise:
            layer_distance = 0.5  # deterministic
            angle_noises = [0, 0, 0]
            z_noises = [0, 0, 0]
            radii = [1.0, 1.0, 1.0]
        else:
            layer_distance = random.uniform(0.3, 2.0)
            angle_noises = [random.uniform(-math.pi / 4, math.pi / 4) for _ in range(3)]
            z_noises = [random.uniform(-0.1, 0.1) for _ in range(3)]
            radii = [random.uniform(0.1, 1.0) for _ in range(3)]

        # Move down the z-axis for this layer
        z_layer -= layer_distance
        # Append 3 new points for this layer
        for angle, angle_noise, radius, z_noise in zip(
            base_angles, angle_noises, radii, z_noises
        ):
            final_angle = angle + angle_noise
            x = math.cos(final_angle) * radius
            y = math.sin(final_angle) * radius
            z = z_layer + z_noise
            positions.append([x, y, z])

    # Step 8: Create edge connections
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

    # Step 9: Assign chirality tags
    chirality = [chirality_value] + [0] * (3 * chirality_distance)
    chirality_tags = [chirality_tag] + [[1, 0, 0]] * (3 * chirality_distance)
    chirality_strs = [[chirality_str] + ["N/A"] * (3 * chirality_distance)]

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
    chirality = torch.tensor(chirality, dtype=torch.float).unsqueeze(-1)
    chirality_one_hot = torch.tensor(chirality_tags, dtype=torch.float)
    positions = torch.tensor(positions, dtype=torch.float)

    # Step 11: Create edge index and make sure its a tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Step 12: Apply centering and random rotation to positions
    positions = center_and_rotate_positions(positions)

    # Step 13: Construct PyTorch Geometric data object
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

    # Step 5: Randomly decide clockwise or counterclockwise placement
    clockwise = random.choice([True, False])

    # Step 6: Arrange atoms
    if clockwise:
        # Sort other atoms in ascending order (smallest to largest)
        triplet_atoms.sort()
        chirality_value = 1
        chirality_tag = [0, 1, 0]
        chirality_str = "R"
    else:
        # Sort other atoms in descending order (largest to smallest)
        triplet_atoms.sort(reverse=True)
        chirality_value = 2
        chirality_tag = [0, 0, 1]
        chirality_str = "S"

    # ------------------------------------
    # Step 7: Assign positions with layers
    # ------------------------------------
    base_angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
    positions = []
    z_layer = 1.0

    # Step 7.1: Chiral Center position
    positions.append([0.0, 0.0, z_layer])

    # Step 7.2: Following layer positions
    for _ in range(chirality_distance):
        if not noise:
            layer_distance = 0.5  # deterministic
            angle_noises = [0, 0, 0]
            z_noises = [0, 0, 0]
            radii = [1.0, 1.0, 1.0]
        else:
            layer_distance = random.uniform(0.3, 2.0)
            angle_noises = [random.uniform(-math.pi / 4, math.pi / 4) for _ in range(3)]
            z_noises = [random.uniform(-0.1, 0.1) for _ in range(3)]
            radii = [random.uniform(0.1, 1.0) for _ in range(3)]

        # Move down the z-axis for this layer
        z_layer -= layer_distance
        # Append 3 new points for this layer
        for angle, angle_noise, radius, z_noise in zip(
            base_angles, angle_noises, radii, z_noises
        ):
            final_angle = angle + angle_noise
            x = math.cos(final_angle) * radius
            y = math.sin(final_angle) * radius
            z = z_layer + z_noise
            positions.append([x, y, z])

    # Step 8: Create edge connections
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

    # Step 9: Assign chirality tags
    chirality = [chirality_value] + [0] * (3 * chirality_distance)
    chirality_tags = [chirality_tag] + [[1, 0, 0]] * (3 * chirality_distance)
    chirality_strs = [chirality_str] + ["N/A"] * (3 * chirality_distance)

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
    chirality = torch.tensor(chirality, dtype=torch.float).unsqueeze(-1)
    chirality_one_hot = torch.tensor(chirality_tags, dtype=torch.float)
    positions = torch.tensor(positions, dtype=torch.float)

    # Step 11: Create edge index and make sure its a tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Step 12: Apply centering and random rotation to positions
    positions = center_and_rotate_positions(positions)

    # Step 13: Construct PyTorch Geometric data object
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


# NOTE Pure chiral configurations based on point clouds and all quadruplets
def create_pure_chiral_instance(points=4, species_range=10):
    ######## Pure Chiral Configuration Point Clouds ########
    """
    Pure Chiral Configuration Point Cloud does not dictate what is a chiral center or what has an edge.
    Instead, it generates a point cloud with the specified number of points and species range.
    The chirality target looks at the handedness of all triplets of points connected to each node.
    """

    # Step 0: Assert necessary conditions
    assert (
        points == 4 or points == 5
    ), "Pure chiral configuration point clouds currently only support 4 or 5 points"
    assert species_range >= 1, "Species range must be at least 1"

    # Step 1: Choose random atomic numbers with replacement
    atomic_numbers = random.choices(range(1, species_range + 1), k=points)
    atomic_numbers = np.array(atomic_numbers)
    atomic_numbers.sort()

    # Step 2: Assign positions randomly between -1 and 1 in 3D space
    positions = np.random.uniform(-1, 1, size=(points, 3))

    # Step 3: For each node, create all triplets involving that node
    chirality = []
    chirality_tags = []
    chirality_strs = []

    for node in range(points):
        # Get all other nodes
        other_nodes = [i for i in range(points) if i != node]
        # Generate all combinations of two other nodes
        triplets = list(combinations(other_nodes, 3))

        # List to store chirality for the current node
        node_chirality = []
        node_chirality_tags = []
        node_chirality_strs = []

        for triplet in triplets:
            triplet = list(triplet)
            # Get atomic numbers
            quadruplet = [node] + triplet
            quadruplet_atomic_numbers = atomic_numbers[quadruplet]

            # Check for unique atomic numbers. If there are repeats, this will not
            # be a chiral configuration
            if len(set(quadruplet_atomic_numbers)) < 4:
                # Not all atomic numbers are unique; assign 'N/A'
                chiral_value = 0
                chiral_tag = [1, 0, 0]  # N/A
                chiral_str = "N/A"
            else:
                # Proceed with chirality computation
                # NOTE atomic numbers are already sorted in ascending order and combinations
                # are generated in lexical order, so we don't need to sort the atomic numbers

                # Compute vectors relative to the first point in the triplet
                # Use Cahn-Ingold-Prelog rules to determine order
                v1 = positions[triplet[2]] - positions[node]
                v2 = positions[triplet[1]] - positions[node]
                v3 = positions[triplet[0]] - positions[node]

                # Compute scalar triple product
                stp = np.dot(v1, np.cross(v2, v3))

                # Determine chirality based on the sign of the scalar triple product
                if stp > 0:
                    chiral_value = 1
                    chiral_tag = [0, 1, 0]  # R
                    chiral_str = "R"
                elif stp < 0:
                    chiral_value = 2
                    chiral_tag = [0, 0, 1]  # S
                    chiral_str = "S"
                else:
                    chiral_value = 0
                    chiral_tag = [1, 0, 0]  # N/A
                    chiral_str = "N/A"

            node_chirality.append(chiral_value)
            node_chirality_tags.append(chiral_tag)
            node_chirality_strs.append(chiral_str)

        chirality.append(node_chirality)
        chirality_tags.append(node_chirality_tags)
        chirality_strs.append(node_chirality_strs)
    # Make tensors
    chirality = torch.tensor(chirality, dtype=torch.float)
    chirality_tags = torch.tensor(chirality_tags, dtype=torch.float)

    # Step 4: Create node features
    atomic_numbers_tensor = torch.tensor(atomic_numbers, dtype=torch.int64).view(-1, 1)
    atomic_numbers_one_hot = F.one_hot(
        atomic_numbers_tensor.squeeze() - 1, num_classes=118
    ).float()

    # Step 5: Create positions tensor
    positions = torch.tensor(positions, dtype=torch.float)

    # NOTE: Edge index is not defined for point clouds
    edge_index = None

    # Step 6: Apply centering and random rotation to positions
    positions = center_and_rotate_positions(positions)

    # Step 7: Create the Data object
    data = Data(
        x=atomic_numbers_one_hot.float(),
        y=chirality,
        z=atomic_numbers_tensor.float(),
        atomic_numbers=atomic_numbers_tensor,
        atomic_numbers_one_hot=atomic_numbers_one_hot,
        chirality=chirality,
        chirality_one_hot=chirality_tags,
        chirality_str=chirality_strs,
        edge_index=edge_index,
        pos=positions,
    )

    return data


def create_chiral_instance(type, chirality_distance, species_range, points, noise):
    assert (
        species_range <= 118
    ), "Species range must be less than or equal to 118 (number of elements in the periodic table)"

    if type == "classic":
        return create_classic_chiral_instance(
            chirality_distance=chirality_distance,
            species_range=species_range,
            noise=noise,
        )
    elif type == "simple":
        return create_simple_chiral_instance(
            chirality_distance=chirality_distance,
            species_range=species_range,
            noise=noise,
        )
    elif type == "crossed":
        return create_crossed_chiral_instance(
            chirality_distance=chirality_distance,
            species_range=species_range,
            noise=noise,
        )
    elif type == "pure":
        return create_pure_chiral_instance(points=points, species_range=species_range)
    else:
        raise ValueError(f"Chiral type not supported: {type}")


def create_dataset(
    num_samples=3000,
    type="simple",
    chirality_distance=1,
    species_range=10,
    points=4,
    save_path="dataset.pt",
    noise=False,
):
    # Communication
    if type == "pure":
        print("Creating dataset with the following parameters:")
        print(f"Number of samples: {num_samples}")
        print(f"Type: {type}")
        print(f"Species range: {species_range}")
        print(f"Points: {points}")
        print(f"Save path: {save_path}")
    else:
        print("Creating dataset with the following parameters:")
        print(f"Number of samples: {num_samples}")
        print(f"Type: {type}")
        print(f"Chirality distance: {chirality_distance}")
        print(f"Species range: {species_range}")
        print(f"Noise: {noise}")
        print(f"Save path: {save_path}")

    # Create
    data_list = []
    for _ in range(num_samples):
        # Generate chiral graph
        data = create_chiral_instance(
            type, chirality_distance, species_range, points, noise
        )
        # Append
        data_list.append(data)
    # Save
    if os.path.dirname(save_path):  # Check if there is a directory in the path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data_list, save_path)
    print(f"Dataset saved as {save_path}")


def debug_exploit_last_three():
    """
    For each dist in [1..3] and type in [simple, crossed, classic], generate 50 samples.

    We then check if the chirality of the last 3 nodes can be determined by the Priority-Based Triple-Product.
    """

    chirality_types = ["simple", "crossed", "classic"]
    num_samples = 50
    species_range = 15  # large enough for dist up to 9, if desired

    for dist in range(1, 4):
        for ctype in chirality_types:
            matches = 0

            for _ in range(num_samples):
                # Generate a chiral instance (no noise for clarity)
                data = create_chiral_instance(
                    type=ctype,
                    chirality_distance=dist,
                    species_range=species_range,
                    points=4,
                    noise=True,
                )

                # ----------- Get CIP Ordering Among the Last 3 Node -----------
                # NOTE it's hardcoded in sample creation that the last 3 nodes are the determining substituents
                last3_z = data.atomic_numbers[-3:].flatten().tolist()  # e.g. [1, 3, 2]
                priority_indices = sorted(
                    range(3), key=lambda i: last3_z[i], reverse=True
                )  # e.g. [1, 2, 0]

                # ----------- (2) PRIORITY-BASED TRIPLE-PRODUCT -----------
                # Get positions in priority order
                last_3_pos = data.pos[-3:]
                pos_ordered = [last_3_pos[i] for i in priority_indices]

                # Do STP calculation
                center_pos = data.pos[0]
                vA = pos_ordered[0] - center_pos
                vB = pos_ordered[1] - center_pos
                vC = pos_ordered[2] - center_pos
                stp = torch.dot(vA, torch.cross(vB, vC))

                # By convention here, let's say: STP>0 => label=1 (R), STP<0 => label=2 (S).
                observed_label = 1 if stp > 0 else 2
                if observed_label == int(data.chirality[0].item()):  # 1=R, 2=S
                    matches += 1

            print(f"Dist={dist}, Type={ctype}: {matches}/{num_samples} matched")


def main():
    # Read Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    args = load_dataset_json(config_path)

    # Extract arguments
    num_samples = args["num_samples"]
    type = args["type"]
    chirality_distance = args["chirality_distance"]
    species_range = args["species_range"]
    points = args["points"]
    save_path = args["save_path"]
    noise = args["noise"]

    # Create
    print("Creating dataset...")
    create_dataset(
        num_samples=num_samples,
        type=type,
        chirality_distance=chirality_distance,
        species_range=species_range,
        points=points,
        save_path=save_path,
        noise=noise,
    )
    print("Dataset created.")


if __name__ == "__main__":
    main()
