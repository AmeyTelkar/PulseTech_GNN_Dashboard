"""
PediatricADRHAN — Heterogeneous Attention Network for Drug-ADR link prediction.
"""
import torch
import torch.nn.functional as F

try:
    from torch_geometric.nn import HANConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class PediatricADRHAN(torch.nn.Module):
    """
    Heterogeneous Attention Network for Drug–ADR link prediction.
    Uses HANConv from PyTorch Geometric.

    Architecture:
    - 2 HANConv layers with multi-head attention
    - ELU activation between layers
    - Dot-product decoder for link prediction
    """

    def __init__(self, in_channels: dict, hidden_channels: int, out_channels: int,
                 metadata, num_heads: int = 4):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required")

        self.conv1 = HANConv(in_channels, hidden_channels, metadata,
                             heads=num_heads, dropout=0.3)
        self.conv2 = HANConv(hidden_channels, out_channels, metadata,
                             heads=1, dropout=0.1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        # HANConv returns None for node types that receive no messages
        # (e.g. after RandomLinkSplit drops some edge types).
        # Filter them out and also prune edges referencing missing types.
        x_dict = {k: F.elu(v) for k, v in x_dict.items() if v is not None}
        available = set(x_dict.keys())
        filtered_edges = {
            k: v for k, v in edge_index_dict.items()
            if k[0] in available and k[2] in available
        }
        x_dict = self.conv2(x_dict, filtered_edges)
        x_dict = {k: v for k, v in x_dict.items() if v is not None}
        return x_dict

    def decode(self, drug_emb, adr_emb, edge_label_index):
        """Decode specific edges for training."""
        src = drug_emb[edge_label_index[0]]
        dst = adr_emb[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def decode_all(self, drug_emb, adr_emb):
        """Decode all possible drug-ADR pairs."""
        return torch.sigmoid(drug_emb @ adr_emb.t())
