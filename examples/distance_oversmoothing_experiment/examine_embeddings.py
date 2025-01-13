import os
import functools

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# E3NN
from e3nn import o3

# Custom
from chi_geometry.dataset import load_dataset_json
from chi_geometry.model import Network, load_model_json, process_batch


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
def compute_oversmoothing_metrics_graph(embs):
    """
    'embs': Tensor [N, D] of node embeddings for a single graph.

    We do:
      1) L2 normalize each embedding => shape [N, D]
      2) Compute average dimension-wise variance
      3) Use matrix multiplication to get pairwise cosine similarities,
         exclude the diagonal, and average over the upper triangle
    """
    N = embs.size(0)
    if N <= 1:
        # If there's only one (or zero) nodes, no pairs to compare
        return {"mean_variance": 0.0, "avg_cos_sim": 0.0}

    # 1) L2 normalize
    embs_normed = F.normalize(embs, p=2, dim=1)

    # 2) Dimension-wise variance => average => scalar
    var_per_dim = embs_normed.var(dim=0, unbiased=False)
    mean_var = var_per_dim.mean().item()

    # 3) Pairwise cosine similarity via matrix multiply
    #    embs_normed @ embs_normed^T => shape [N, N]
    cos_matrix = embs_normed @ embs_normed.t()

    # exclude diagonal => number of unique pairs => upper triangle
    mask = torch.ones(N, N, device=cos_matrix.device).triu(diagonal=1).bool()
    avg_cos_sim = cos_matrix[mask].mean().item()  # average of upper-tri off-diagonal

    return {"mean_variance": mean_var, "avg_cos_sim": avg_cos_sim}


def compute_oversmoothing_metrics_dataset(
    loader, model, device, process_batch, activations
):
    total_var = 0.0
    total_cos = 0.0
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
            metrics = compute_oversmoothing_metrics_graph(node_embs)
            total_var += metrics["mean_variance"]
            total_cos += metrics["avg_cos_sim"]
            graph_count += 1

    if graph_count == 0:
        return {"mean_variance": 0.0, "avg_cos_sim": 0.0}

    avg_var = total_var / graph_count
    avg_cos = total_cos / graph_count
    return {"mean_variance": avg_var, "avg_cos_sim": avg_cos}


###############################################################################
# 3) Main analysis function that loads model, data, & measures oversmoothing
###############################################################################
def analyze_oversmoothing(distance):
    """
    1. Load configs
    2. Load dataset for 'distance'
    3. Load best model, attach forward hook on final conv
    4. For each split (train, val, test), run each graph individually (batch_size=1),
       compute oversmoothing metrics for that single graph, accumulate, then average.
    5. Write results to smoothness.txt
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---------------------------
    # 1) Read dataset
    # ---------------------------
    dataset_config_path = os.path.join(script_dir, "dataset_config.json")
    dataset_args = load_dataset_json(dataset_config_path)
    num_samples = dataset_args["num_samples"]
    dataset_path = os.path.join(
        f"datasets/{num_samples}-samples_{distance}-distance.pt"
    )
    dataset = torch.load(dataset_path)

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
    # 2) Construct model architecture
    # ---------------------------
    model_config_path = os.path.join(script_dir, "model_config.json")
    model_args = load_model_json(model_config_path)
    device = torch.device(
        "cuda" if model_args["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    layers = distance + 3
    num_classes = model_args["num_classes"]

    model = Network(
        irreps_in=o3.Irreps(model_args["irreps_in"]),
        irreps_hidden=o3.Irreps(model_args["irreps_hidden"]),
        irreps_out=o3.Irreps(model_args["irreps_out"]),
        irreps_node_attr=o3.Irreps(model_args["irreps_node_attr"]),
        irreps_edge_attr=o3.Irreps(model_args["irreps_edge_attr"]),
        layers=layers,
        max_radius=model_args["max_radius"],
        number_of_basis=model_args["number_of_basis"],
        radial_layers=model_args["radial_layers"],
        radial_neurons=model_args["radial_neurons"],
        num_neighbors=model_args["num_neighbors"],
        num_nodes=model_args["num_nodes"],
        reduce_output=model_args["reduce_output"],
        num_classes=num_classes,
        num_heads=model_args["num_heads"],
    ).to(device)

    # Load best checkpoint
    run_dir = os.path.join(script_dir, f"logs/{distance}-distance")
    best_model_path = os.path.join(run_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # ---------------------------
    # 3) Attach forward hook to final conv
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
    # 5) Print + Write to smoothness.txt
    # ---------------------------
    print(f"\n=== Oversmoothing Analysis for distance={distance} ===")
    print(f"Train -> {train_metrics}")
    print(f"Val   -> {val_metrics}")
    print(f"Test  -> {test_metrics}")
    print("-----------------------------------------------------")

    os.makedirs(run_dir, exist_ok=True)
    smoothness_path = os.path.join(run_dir, "smoothness.txt")
    with open(smoothness_path, "w") as f:
        f.write("===== Oversmoothing Analysis (per-graph averaging) =====\n")
        f.write(f"Distance: {distance}\n\n")

        f.write("Train:\n")
        f.write(f"  Mean Variance: {train_metrics['mean_variance']:.4f}\n")
        f.write(f"  Avg Cosine Sim: {train_metrics['avg_cos_sim']:.4f}\n\n")

        f.write("Val:\n")
        f.write(f"  Mean Variance: {val_metrics['mean_variance']:.4f}\n")
        f.write(f"  Avg Cosine Sim: {val_metrics['avg_cos_sim']:.4f}\n\n")

        f.write("Test:\n")
        f.write(f"  Mean Variance: {test_metrics['mean_variance']:.4f}\n")
        f.write(f"  Avg Cosine Sim: {test_metrics['avg_cos_sim']:.4f}\n\n")

    print(f"Oversmoothing metrics written to {smoothness_path}")


###############################################################################
# 6) Main
###############################################################################
if __name__ == "__main__":
    distances = [3]
    for dist in distances:
        analyze_oversmoothing(dist)
