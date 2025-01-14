# General
import os

# Torch
import torch

# Visualization
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Custom
from chi_geometry.dataset import load_dataset_json


def plot_graph(dataset_path="dataset.pt", cmap="viridis"):
    # Some Favorite Color Maps: 'viridis', 'Spectral'
    # Load dataset and select the first molecule/graph
    dataset = torch.load(dataset_path)
    graph = dataset[0]

    # Get node positions and atomic numbers
    positions = graph.pos.numpy()
    atomic_numbers = graph.z.numpy()

    # Create color map based on atomic numbers (scaled to a colormap range)
    cmap = plt.colormaps[cmap]
    norm = Normalize(vmin=atomic_numbers.min(), vmax=atomic_numbers.max())
    colors = cmap(norm(atomic_numbers))

    # Plot the 3D structure with matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    # ax.set_axis_off()  # Hide axes

    # Plot nodes with colors corresponding to atomic numbers
    for pos, color in zip(positions, colors):
        ax.scatter(*pos, color=color, s=100, edgecolors="k", alpha=0.9)

    # Draw edges between nodes based on edge_index
    edge_index = graph.edge_index
    if edge_index is not None:
        edge_index = edge_index.numpy()
        for start, end in edge_index.T:
            start_pos = positions[start]
            end_pos = positions[end]
            ax.plot(*zip(start_pos, end_pos), color="gray", alpha=0.7)

    # Set plot parameters
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(
        f"Graph Visualization | Chiral Center has Label: {graph.chirality_str[0][0]}"
    )

    # Add color bar for atomic numbers
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Node Species")

    # Show the plot (interactive)
    plt.show()


def main():
    # Read Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    args = load_dataset_json(config_path)

    # Plot
    plot_graph(dataset_path=args["save_path"], cmap=args["cmap"])


if __name__ == "__main__":
    main()
