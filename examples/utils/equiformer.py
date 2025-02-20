# General

# Torch
import torch
import torch.nn.utils.rnn as rnn_utils

# Equivariant Neural Networks
from equiformer_pytorch import Equiformer as OpenSourceEquiformer


############################################################################################################
# This is a model taken from the equiformer library by lucidrainsand modified to work with the Chi-Geometry dataset.
# The original model can be found here: https://github.com/lucidrains/equiformer-pytorch
# Accessed: Jan 14th, 2025
############################################################################################################
class Equiformer(torch.nn.Module):
    """
    A simple Equiformer-based model for classification.
    Uses three degrees (l=0,1,2) with specified hidden dimensions,
    then maps the degree-0 output to 'num_classes'.

    There is reshaping and masking logic because transformers expect
    a batch dimension with constant graph size.
    """

    def __init__(self, input_dim, hidden_dim, layers, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.output_dim = output_dim

        self.input_embedding = torch.nn.Linear(
            118, self.hidden_dim
        )  # Embed one-hot atom type
        # Equiformer setup:
        #  - degree-0 dimension = hidden_dim
        #  - degree-1 dimension = hidden_dim//2
        #  - degree-2 dimension = hidden_dim//8
        self.equiformer = OpenSourceEquiformer(
            dim=(self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 8),
            num_degrees=3,  # l=0,1,2
            depth=self.layers,
            num_linear_attn_heads=1,
            reduce_dim_out=False,
            attend_self=True,
            heads=(4, 1, 1),
            dim_head=(self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 8),
        )

        self.out_embed = torch.nn.Linear(
            self.hidden_dim, self.output_dim
        )  # Map degree-0 output to desired prediction dimension

    def reshape_batch(
        self, feats: torch.Tensor, coords: torch.Tensor, batch: torch.Tensor
    ):
        """
        feats:  [total_nodes, feature_dim]  (concatenated from multiple graphs)
        coords: [total_nodes, 3]           (xyz positions for each node)
        batch:  [total_nodes]              (which graph each node belongs to, in range [0..B-1])

        Returns:
        feats_3d:  [B, N_max, feature_dim]
        coords_3d: [B, N_max, 3]
        mask:      [B, N_max] (True where node is real, False where padded)
        """
        device = feats.device

        # 1) Figure out how many graphs we have
        num_graphs = batch.max().item() + 1

        # 2) Split feats and coords by graph
        feats_per_graph = []
        coords_per_graph = []
        lengths = []
        for g in range(num_graphs):
            node_indices = (batch == g).nonzero(as_tuple=True)[0]
            feats_per_graph.append(feats[node_indices])  # shape [n_g, feature_dim]
            coords_per_graph.append(coords[node_indices])  # shape [n_g, 3]
            lengths.append(len(node_indices))  # store how many nodes in this graph

        # 3) Pad each list to get a uniform node dimension = N_max
        #    feats_3d => [B, N_max, feature_dim]
        #    coords_3d => [B, N_max, 3]
        feats_3d = rnn_utils.pad_sequence(
            feats_per_graph, batch_first=True, padding_value=0.0
        )
        coords_3d = rnn_utils.pad_sequence(
            coords_per_graph, batch_first=True, padding_value=0.0
        )

        # 4) Build a boolean mask of shape [B, N_max], marking real vs. padded
        N_max = feats_3d.size(1)
        mask = torch.zeros((num_graphs, N_max), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        return feats_3d.to(device), coords_3d.to(device), mask

    def forward(self, data):
        # 0) Unpack data
        feats, coords, mask = self.reshape_batch(data.x, data.pos, data.batch)
        # 1) Embed down to hidden_dim
        feats = self.input_embedding(feats)  # [B, N, input_dim]
        # 2) Equiformer
        out = self.equiformer(
            feats,
            coords,
            mask=mask,  # shape [B, N]
        )

        # 3) Use degree-0 outputs for classification
        node_embs = out.type0  # [B, N, hidden_dim]

        # 4) Reshape and mask to expected node outputs
        node_embs = node_embs.view(-1, self.hidden_dim)  # [B*N, output_dim]
        node_embs = node_embs[mask.view(-1)]  # [total_valid_nodes, hidden_dim]

        # 5) Map to output_dim
        out = self.out_embed(node_embs)

        return out
