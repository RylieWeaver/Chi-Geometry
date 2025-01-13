import torch
from torch_geometric.data import Data


def make_global_connections(dataset):
    for data in dataset:
        num_nodes = data.num_nodes
        edge_pairs = []

        # Get global connections
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_pairs.append([i, j])

        # Overwrite
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        data.edge_index = edge_index

    return dataset
