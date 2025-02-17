# General
import os

# Torch
import torch

# Visualization
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator

# Custom
from chi_geometry import load_dataset_json

# NOTE I've changed some stuff in the plotting to make figures nicely in this branch


def plot_graph(graph, cmap="viridis"):
    # Some Favorite Color Maps: 'viridis', 'Spectral'

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

    # Mirror
    # positions = positions[[0,1,2,4,3], :]

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

    # -----------------------
    # Node labeling
    # -----------------------
    # First node is labeled "c"
    labels = [None] * 5
    labels[0] = "c"

    # Sort the other 4 by descending atomic_number and label them "1", "2", "3", "4"
    remaining_indices = [1, 2, 3, 4]
    sorted_indices = sorted(
        remaining_indices, key=lambda i: atomic_numbers[i], reverse=True
    )
    for rank, idx in enumerate(sorted_indices, start=1):
        labels[idx] = str(rank)

    # Place labels near each node
    for i, (x, y, z) in enumerate(positions):
        ax.text(
            x + 0.05,
            y + 0.05,
            z + 0.05,
            labels[i],
            color="black",
            fontsize=10,
            zorder=5,
        )

    # Set plot parameters
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    ax.zaxis.set_major_locator(MultipleLocator(0.2))
    plt.title(
        f"Graph Visualization | Chiral Center has Label: {graph.chirality_str[0][0]}"
    )

    # Add color bar for atomic numbers
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Node Species")

    # Save the plot
    plt.savefig("graph_plot.png", dpi=300, bbox_inches="tight")

    # Show the plot (interactive)
    plt.show()


def main():
    # Read Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    args = load_dataset_json(config_path)

    # Load dataset and select the first molecule/graph
    dataset = torch.load(args["save_path"], weights_only=False)
    graph = dataset[0]

    # Plot
    plot_graph(graph, cmap=args["cmap"])


if __name__ == "__main__":
    main()
