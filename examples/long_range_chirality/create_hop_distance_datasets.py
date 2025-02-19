# General
import os

# Chi-Geometry
from experiment_utils.utils import create_hop_distance_datasets


def main():
    # Setup
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    distances = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Create datasets
    dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    create_hop_distance_datasets(distances, dataset_config_path)


if __name__ == "__main__":
    main()
