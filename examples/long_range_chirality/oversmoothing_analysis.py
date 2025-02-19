# General
import os
import functools

# Torch
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# E3NN
from e3nn import o3

# Custom
from examples.utils import Network, load_model_json, process_batch


###############################################################################
# 1) Hook function to capture final conv outputs
###############################################################################
def store_activations(activations_dict, module, input, output):
    """
    PyTorch forward-hook callback.
    'output' is the final conv layer's output embeddings for all nodes in a single graph
    (when batch_size=1), or multiple graphs if batch_size>1.
    We store them in a dictionary so we can retrieve them later.
    """
    activations_dict["final_conv_output"] = output


###############################################################################
# 2) COMPUTING OVERSMOOTHING METRICS FUNCTIONS
###############################################################################
def compute_dirichlet_energy(embs: torch.Tensor, edge_index: torch.Tensor) -> float:
    """
    embs:       shape [N, D], node embeddings
    edge_index: shape [2, E], edges = (row[], col[])

    Returns the Dirichlet energy:
        sqrt( (1/v) * sum_{(i,j) in edges} ||embs[i] - embs[j]||^2 )
    """
    i, j = edge_index  # row & col of edges
    # Differences for each edge => shape [2E, D] (double edges in edge_index)
    diffs = embs[i] - embs[j]
    # L2 squared => shape [E]
    sq_dists = (diffs**2).sum(dim=-1) / 2  # Divide by 2 to avoid double counting

    v = embs.size(0)  # number of nodes
    sum_sq = sq_dists.sum()
    dirichlet_value = (sum_sq / v).sqrt().item()
    return dirichlet_value


# NOTE Dirichlet energy seems to be the most respected and reliable metric for oversmoothing
def compute_oversmoothing_metrics_graph(embs, edge_index):
    """
    Now includes Dirichlet energy as recommended.
    Also keeps your original mean_variance & avg_cos_sim if you want them.
    """
    N = embs.size(0)
    # If fewer than 2 nodes => trivially zero or undefined
    if N <= 1:
        return {"mean_variance": 0.0, "avg_cos_sim": 0.0, "dirichlet": 0.0}

    # 1) L2 normalize (your original step)
    embs_normed = F.normalize(embs, p=2, dim=1)

    # 2) Dimension-wise variance => average => scalar
    var_per_dim = embs_normed.var(dim=0, unbiased=False)
    mean_var = var_per_dim.mean().item()

    # 3) Pairwise cosine similarity
    cos_matrix = embs_normed @ embs_normed.t()
    mask = torch.ones(N, N, device=cos_matrix.device).triu(diagonal=1).bool()
    avg_cos_sim = cos_matrix[mask].mean().item()

    # 4) Dirichlet energy
    dirichlet_val = compute_dirichlet_energy(embs, edge_index)

    return {
        "mean_variance": mean_var,
        "avg_cos_sim": avg_cos_sim,
        "dirichlet": dirichlet_val,
    }


def compute_oversmoothing_metrics_dataset(
    loader, model, device, process_batch, activations
):
    total_var = 0.0
    total_cos = 0.0
    total_dirichlet = 0.0
    graph_count = 0

    with torch.no_grad():
        for single_graph_data in loader:
            single_graph_data = single_graph_data.to(device)
            single_graph_data = process_batch(single_graph_data)

            # Forward pass on this single graph
            _ = model(single_graph_data)

            # final_conv_output: shape [num_nodes_in_graph, out_dim]
            node_embs = activations["final_conv_output"].cpu()

            # Compute oversmoothing metrics for this single graph
            metrics = compute_oversmoothing_metrics_graph(
                node_embs, single_graph_data.edge_index  # needed for Dirichlet
            )
            total_var += metrics["mean_variance"]
            total_cos += metrics["avg_cos_sim"]
            total_dirichlet += metrics["dirichlet"]
            graph_count += 1

    if graph_count == 0:
        return {"mean_variance": 0.0, "avg_cos_sim": 0.0, "dirichlet": 0.0}

    return {
        "mean_variance": total_var / graph_count,
        "avg_cos_sim": total_cos / graph_count,
        "dirichlet": total_dirichlet / graph_count,
    }


###############################################################################
# 3) Main analysis function that computes oversmoothing metrics
###############################################################################
def analyze_oversmoothing(model, dataset, distance, log_dir, device="cpu"):
    # ---------------------------
    # 1) Load dataset
    # ---------------------------
    # same shuffle as training
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(dataset))
    shuffled_dataset = [dataset[i] for i in shuffled_indices]

    train_size = int(0.8 * len(shuffled_dataset))
    val_size = int(0.1 * len(shuffled_dataset))
    train_dataset = shuffled_dataset[:train_size]
    val_dataset = shuffled_dataset[train_size : train_size + val_size]
    test_dataset = shuffled_dataset[train_size + val_size :]

    # Build data loaders, with batch_size=1 to handle one graph at a time
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ---------------------------
    # 2) Attach forward hook to final conv of model
    # ---------------------------
    activations = {}
    hook_handle = model.layers[-1].register_forward_hook(
        functools.partial(store_activations, activations)
    )

    train_metrics = compute_oversmoothing_metrics_dataset(
        train_loader, model, device, process_batch, activations
    )
    val_metrics = compute_oversmoothing_metrics_dataset(
        val_loader, model, device, process_batch, activations
    )
    test_metrics = compute_oversmoothing_metrics_dataset(
        test_loader, model, device, process_batch, activations
    )

    hook_handle.remove()  # done hooking

    # ---------------------------
    # 3) Print + Write to smoothness.txt
    # ---------------------------
    print(f"\n=== Oversmoothing Analysis for distance={distance} ===")
    print(f"Train -> {train_metrics}")
    print(f"Val   -> {val_metrics}")
    print(f"Test  -> {test_metrics}")
    print("-----------------------------------------------------")

    os.makedirs(log_dir, exist_ok=True)
    smoothness_path = os.path.join(log_dir, "smoothness.txt")
    with open(smoothness_path, "w") as f:
        f.write("===== Oversmoothing Analysis (per-graph averaging) =====\n")
        f.write(f"Distance: {distance}\n\n")

        f.write("Train:\n")
        f.write(f"  Mean Variance: {train_metrics['mean_variance']:.4f}\n")
        f.write(f"  Avg Cosine Sim: {train_metrics['avg_cos_sim']:.4f}\n")
        f.write(f"  Sqrt Dirichlet Energy: {train_metrics['dirichlet']:.4f}\n\n")

        f.write("Val:\n")
        f.write(f"  Mean Variance: {val_metrics['mean_variance']:.4f}\n")
        f.write(f"  Avg Cosine Sim: {val_metrics['avg_cos_sim']:.4f}\n")
        f.write(f"  Sqrt Dirichlet Energy: {val_metrics['dirichlet']:.4f}\n\n")

        f.write("Test:\n")
        f.write(f"  Mean Variance: {test_metrics['mean_variance']:.4f}\n")
        f.write(f"  Avg Cosine Sim: {test_metrics['avg_cos_sim']:.4f}\n")
        f.write(f"  Sqrt Dirichlet Energy: {test_metrics['dirichlet']:.4f}\n\n")

    print(f"Oversmoothing metrics written to {smoothness_path}")


###############################################################################
# 4) Main
###############################################################################
def main():
    # Setup
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    distances = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Create datasets
    # dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    # create_all_datasets(distances, dataset_config_path)

    # Args
    model_config_path = os.path.join(script_dir, "model_config.json")
    model_args = load_model_json(model_config_path)
    noise = False
    datadir = "datasets"
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    # Train model on each dataset
    for dist in distances:
        # Dataset
        if noise:
            dataset_path = os.path.join(f"{datadir}/noise-{dist}-distance/dataset.pt")
        else:
            dataset_path = os.path.join(f"{datadir}/{dist}-distance/dataset.pt")
        dataset = torch.load(dataset_path)
        print(f"Dataset contains {len(dataset)} graphs.")

        # Model
        modelname = f"local_e3nn"
        if noise:
            log_dir = f"logs/noise-{dist}-distance-{modelname}"
        else:
            log_dir = f"logs/{dist}-distance-{modelname}"
        model_args["layers"] = (
            dist + 3
        )  # Increase layers with distance to ensure propagation of chiral information
        model = Network(
            irreps_in=o3.Irreps(model_args["irreps_in"]),
            irreps_hidden=o3.Irreps(model_args["irreps_hidden"]),
            irreps_out=o3.Irreps(model_args["irreps_out"]),
            irreps_node_attr=o3.Irreps(model_args["irreps_node_attr"]),
            irreps_edge_attr=o3.Irreps(model_args["irreps_edge_attr"]),
            layers=model_args["layers"],
            max_radius=model_args["max_radius"],
            number_of_basis=model_args["number_of_basis"],
            radial_layers=model_args["radial_layers"],
            radial_neurons=model_args["radial_neurons"],
            num_neighbors=model_args["num_neighbors"],
            num_nodes=model_args["num_nodes"],
            reduce_output=model_args["reduce_output"],
            num_classes=model_args["num_classes"],
            num_heads=model_args["num_heads"],
        ).to(device)
        model.load_state_dict(torch.load(f"{log_dir}/best_model.pt"))
        model.eval()
        print(f"Training model for distance {dist}...")
        analyze_oversmoothing(model, dataset, dist, log_dir, device)

    print("Experiment complete.")


if __name__ == "__main__":
    main()


###############################################################################
# 7) Notes
###############################################################################
# Epochs to converge to perfect accuracy on train/val/test by distance:
#  - distance=1:  32 epochs
#  - distance=2:  77 epochs
#  - distance=3:  261 epochs
#  - distance=4:  343 epochs
#  - distance=5:  N/A  (carried out 492 epochs)
#  - distance=6:  N/A  (early stopped at 76 epochs)
#  - distance=7:  N/A  (early stopped at 76 epochs)
#  - distance=8:  N/A  (early stopped at 76 epochs)
#  - distance=9:  N/A  (early stopped at 75 epochs) (nans in loss)
# 500 epochs and 75 early stopping

# Did not find any clear oversmoothing trends dirichlet energy and weak evidence in other metrics.
