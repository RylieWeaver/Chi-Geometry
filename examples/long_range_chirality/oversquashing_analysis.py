import os
import functools
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# E3NN
from e3nn import o3

# Custom imports
from examples.utils import Network, load_model_json, process_batch


###############################################################################
# 1) Helper: BFS to find hop distances
###############################################################################
def compute_node_distances(edge_index, num_nodes, center_idx):
    """
    Simple BFS to find the hop distance of each node from `center_idx`.
    Returns an array `distances` of length num_nodes,
    where distances[i] is the hop-distance from center_idx to node i,
    or -1 if unreachable.
    """
    adj_list = [[] for _ in range(num_nodes)]
    i, j = edge_index
    for u, v in zip(i.tolist(), j.tolist()):
        adj_list[u].append(v)
        adj_list[v].append(u)

    distances = [-1] * num_nodes
    distances[center_idx] = 0
    queue = deque([center_idx])
    while queue:
        cur = queue.popleft()
        for nbr in adj_list[cur]:
            if distances[nbr] == -1:
                distances[nbr] = distances[cur] + 1
                queue.append(nbr)

    return distances


###############################################################################
# 2) Gradient-based Over-Squashing Test
###############################################################################
def gradient_test_chiral_center(model, data, chiral_idx, device):
    """
    1) data.atomic_numbers_one_hot.requires_grad_()
    2) Single-node classification loss at chiral_idx
       (with unsqueezing logic corrected)
    3) BFS -> bucket grad norms by distance
    Returns {distance: average_grad_norm_in_this_graph_for_that_distance}.
    """

    data = data.to(device)
    data.atomic_numbers_one_hot.requires_grad_()

    model.zero_grad()
    if data.atomic_numbers_one_hot.grad is not None:
        data.atomic_numbers_one_hot.grad.zero_()

    # Forward pass => node-level logits => shape [num_nodes, num_classes], but we do [0] for first head
    out = model(data)[0]  # shape [num_nodes, num_classes], from your script

    # The label for the chiral node (assume data.y[chiral_idx] is valid)
    chiral_label = data.y[chiral_idx].long()  # shape []
    chiral_logit = out[chiral_idx].unsqueeze(0)  # shape [1, num_classes]

    criterion = torch.nn.CrossEntropyLoss()
    loss_chiral = criterion(chiral_logit, chiral_label)

    loss_chiral.backward()

    # Grad norms => shape [num_nodes]
    grad_norms = data.atomic_numbers_one_hot.grad.norm(dim=-1).cpu()

    # BFS distances
    distances = compute_node_distances(data.edge_index, data.num_nodes, chiral_idx)

    # Collect node grad norms by BFS distance
    dist_buckets = {}
    for node_i, dist in enumerate(distances):
        if dist < 0:
            continue
        dist_buckets.setdefault(dist, []).append(grad_norms[node_i].item())

    # Convert each distance bucket to an average
    dist_to_avg_grad = {}
    for dist, grad_list in dist_buckets.items():
        dist_to_avg_grad[dist] = sum(grad_list) / len(grad_list)

    return dist_to_avg_grad


###############################################################################
# 3) The Main Oversquashing Script
###############################################################################
def oversquashing_analysis(
    model, dataset, distance, log_dir, max_graphs=1000, device="cpu"
):
    # Shuffle, as in training
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(dataset))
    shuffled_dataset = [dataset[i] for i in shuffled_indices]

    # Build loader
    test_loader = DataLoader(shuffled_dataset, batch_size=1, shuffle=False)

    # We'll store final results in dist_sums and dist_counts
    dist_sums = defaultdict(float)
    dist_counts = defaultdict(int)

    graph_count = 0
    # 3) For each graph, up to max_graphs
    for idx, single_graph_data in enumerate(test_loader):
        if idx >= max_graphs:
            break

        single_graph_data = single_graph_data.to(device)
        single_graph_data = process_batch(single_graph_data)

        # find a node with deg=3
        deg = single_graph_data.edge_index.new_zeros(single_graph_data.num_nodes)
        i, j = single_graph_data.edge_index
        for u, v in zip(i.tolist(), j.tolist()):
            deg[u] += 1
            deg[v] += 1

        chiral_candidates = (deg == 3).nonzero(as_tuple=True)[0].tolist()
        if len(chiral_candidates) == 0:
            chiral_idx = 0
        else:
            chiral_idx = chiral_candidates[0]

        # gradient test
        dist_to_avg_grad = gradient_test_chiral_center(
            model, single_graph_data, chiral_idx, device
        )

        # Accumulate sums, counts
        for dist, avg_grad in dist_to_avg_grad.items():
            dist_sums[dist] += avg_grad
            dist_counts[dist] += 1

        graph_count += 1

    # 4) Compute and write results to oversquashing.txt
    oversq_path = os.path.join(log_dir, "oversquashing.txt")
    os.makedirs(log_dir, exist_ok=True)

    with open(oversq_path, "w") as f_out:
        f_out.write(
            f"=== Oversquashing Analysis (distance={distance}, max_graphs={max_graphs}) ===\n"
        )
        f_out.write(f"{graph_count} graphs processed.\n\n")

        all_dists = sorted(dist_sums.keys())

        # 4.1) Compute each distance's average first
        raw_averages = {}
        for dist in all_dists:
            c = dist_counts[dist]
            if c > 0:
                raw_averages[dist] = dist_sums[dist] / c
            else:
                raw_averages[dist] = 0.0  # or skip

        # 4.2) Find the maximum among these averages (avoid zero division)
        norm = sum(raw_averages.values()) if len(raw_averages) > 0 else 1e-15
        if norm < 1e-15:
            norm = 1e-15  # fallback if they were all 0

        # 4.3) Print + Write normalized results
        for dist in all_dists:
            avg_grad = raw_averages[dist]
            if dist_counts[dist] > 0:
                normalized = avg_grad / norm
                f_out.write(
                    f"Distance {dist} => raw avg: {avg_grad:.4f}, normalized: {normalized:.4f}\n"
                )
            else:
                f_out.write(f"Distance {dist} => no data.\n")

    print(f"Oversquashing results (normalized) written to {oversq_path}")


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
        oversquashing_analysis(model, dataset, dist, log_dir, device=device)

    print("Experiment complete.")


if __name__ == "__main__":
    main()


###############################################################################
# 5) Notes
###############################################################################
# Oversquashing metrics (grad of node features) (normalized):
#  - distance=1:  0.5498
#  - distance=2:  0.5772
#  - distance=3:  0.0512
#  - distance=4:  0.0061
#  - distance=5:  0.0000
#  - distance=6:  nan
#  - distance=7:  nan
#  - distance=8:  nan
#  - distance=9:  nan

# The results do indicate oversquashing after distance=2.
