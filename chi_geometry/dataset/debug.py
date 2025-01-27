# General
import numpy as np

# Torch
import torch

# Chi-Geometry
from chi_geometry.dataset import create_chiral_instance, scalar_triple_product


def check_classic_configurations(num_samples=100, species_range=15, dist_range=9):
    # NOTE These debugs are heavily hardcoded for their specific configuration
    #      and what is expected for the scalar triple productnin any/all of the
    #      layers.

    ctype = "classic"

    # Check number samples for each distance
    for dist in range(1, dist_range + 1):
        matches = 0

        for _ in range(num_samples):
            data = create_chiral_instance(
                type=ctype,
                chirality_distance=dist,
                species_range=species_range,
                points=4,
                noise=False,
            )

            # Get CIP ordering among the last 3 nodes
            last4_z = (
                data.atomic_numbers[-4:].flatten().tolist()
            )  # e.g. [1, 3, 2, 4]  # NOTE: The first should have the lowest priority
            priority_indices = sorted(
                range(4), key=lambda i: last4_z[i], reverse=True
            )  # e.g. [3, 1, 2, 0]

            # Flag to ensure ALL layers have a positive or negative STP
            # The classic type has a simple geometry such that we'd expect
            # this of all the layers.
            all_positive = True
            all_negative = True

            # Simple should have a consistent stp for all layers
            for i in range(dist):
                slice = [i * 4 + 1, (i + 1) * 4 + 1]
                # Get positions for the (i+1)-th layer
                pos_layer = data.pos[slice[0] : slice[1]]

                # Reorder the positions by CIP priority
                pos_ordered = [pos_layer[idx] for idx in priority_indices]

                # Compute STP with the chiral center at index 0
                center_pos = data.pos[0]
                stp = scalar_triple_product(
                    center_pos - pos_ordered[3],
                    pos_ordered[1] - pos_ordered[0],
                    pos_ordered[2] - pos_ordered[1],
                )

                # Check stp is consistent and break if not
                if stp >= 0:
                    all_negative = False
                elif stp <= 0:
                    all_positive = False
                if not all_positive and not all_negative:
                    break

            # Only increment matches if ALL layers had consistent STP
            if (all_positive and data.chirality_str[0][0] == "R") or (
                all_negative and data.chirality_str[0][0] == "S"
            ):
                matches += 1

        print(f"Dist={dist}, Type={ctype}: {matches}/{num_samples} matched")


def check_simple_configurations(num_samples=100, species_range=15, dist_range=9):
    # NOTE These debugs are heavily hardcoded for their specific configuration
    #      and what is expected for the scalar triple productnin any/all of the
    #      layers.

    ctype = "simple"

    # Check number samples for each distance
    for dist in range(1, dist_range + 1):
        matches = 0

        for _ in range(num_samples):
            data = create_chiral_instance(
                type=ctype,
                chirality_distance=dist,
                species_range=species_range,
                points=4,
                noise=True,
            )

            # Get CIP ordering among the last 3 nodes
            last3_z = data.atomic_numbers[-3:].flatten().tolist()  # e.g. [1, 3, 2]
            priority_indices = sorted(
                range(3), key=lambda i: last3_z[i], reverse=True
            )  # e.g. [1, 2, 0]

            # Flag to ensure ALL layers have a positive or negative STP
            # The simple type has a simple geometry such that we'd expect
            # this of all the layers.
            all_positive = True
            all_negative = True

            # Simple should have a consistent stp for all layers
            for i in range(dist):
                slice = [i * 3 + 1, (i + 1) * 3 + 1]
                # Get positions for the (i+1)-th layer
                pos_layer = data.pos[slice[0] : slice[1]]

                # Reorder the positions by CIP priority
                pos_ordered = [pos_layer[idx] for idx in priority_indices]

                # Compute STP with the chiral center at index 0
                center_pos = data.pos[0]
                stp = scalar_triple_product(
                    pos_ordered[0] - center_pos,
                    pos_ordered[1] - center_pos,
                    pos_ordered[2] - center_pos,
                )

                # Check stp is consistent and break if not
                if stp >= 0:
                    all_negative = False
                elif stp <= 0:
                    all_positive = False
                if not all_positive and not all_negative:
                    break

            # Only increment matches if ALL layers had consistent STP
            if (all_positive and data.chirality_str[0][0] == "R") or (
                all_negative and data.chirality_str[0][0] == "S"
            ):
                matches += 1

        print(f"Dist={dist}, Type={ctype}: {matches}/{num_samples} matched")


def check_crossed_configurations(num_samples=100, species_range=15, dist_range=9):
    # NOTE These debugs are heavily hardcoded for their specific configuration
    #      and what is expected for the scalar triple productnin any/all of the
    #      layers.

    ctype = "crossed"

    # Check number samples for each distance
    for dist in range(1, dist_range + 1):
        matches = 0

        for _ in range(num_samples):
            data = create_chiral_instance(
                type=ctype,
                chirality_distance=dist,
                species_range=species_range,
                points=4,
                noise=True,
            )

            # Flag to ensure ALL layers have the stp we expect
            # The ordering/edge are unclear but we do expect that the appended
            # positions are all in clockwise order and hence the STP should be positive
            all_consistent = True

            # Simple should have a consistent stp for all layers
            for i in range(dist - 1):
                slice = [i * 3 + 1, (i + 1) * 3 + 1]
                # Get positions for the (i+1)-th layer
                pos_layer = data.pos[slice[0] : slice[1]]

                # Compute STP with the chiral center at index 0
                center_pos = data.pos[0]
                stp = scalar_triple_product(
                    pos_layer[0] - center_pos,
                    pos_layer[1] - center_pos,
                    pos_layer[2] - center_pos,
                )

                # Check stp is consistent and break if not
                if stp < 0:
                    all_consistent = False
                    break

            # In the last layer, we will know the ordering and should check the STP
            # Get CIP ordering among the last 3 nodes
            last3_z = data.atomic_numbers[-3:].flatten().tolist()  # e.g. [1, 3, 2]
            priority_indices = sorted(
                range(3), key=lambda i: last3_z[i], reverse=True
            )  # e.g. [1, 2, 0]

            # Get positions for the dist-th layer
            slice = [(dist - 1) * 3 + 1, dist * 3 + 1]
            pos_layer = data.pos[slice[0] : slice[1]]

            # Reorder the positions by CIP priority
            pos_ordered = [pos_layer[idx] for idx in priority_indices]

            # Compute STP with the chiral center at index 0
            center_pos = data.pos[0]
            stp = scalar_triple_product(
                pos_ordered[0] - center_pos,
                pos_ordered[1] - center_pos,
                pos_ordered[2] - center_pos,
            )

            # Check stp is consistent and break if not
            if stp < 0 and data.chirality_str[0][0] == "R":
                all_consistent = False
            elif stp > 0 and data.chirality_str[0][0] == "S":
                all_consistent = False
            elif stp == 0:
                all_consistent = False
                print("Scalar triple product is 0")

            # Only increment matches if ALL layers had consistent STP
            if all_consistent:
                matches += 1

        print(f"Dist={dist}, Type={ctype}: {matches}/{num_samples} matched")


if __name__ == "__main__":
    check_classic_configurations()
    check_simple_configurations()
    check_crossed_configurations()
