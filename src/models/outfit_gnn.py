import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv, BatchNorm
except Exception:
    SAGEConv = None


class GNNEncoder(nn.Module):
    """
    Encodes node features into low-dimensional embeddings.

    Args:
        in_dim: input feature dimension
        hidden_dims: list of hidden dims (per conv layer)
        out_dim: final output embedding dim
        use_bn: whether to use batch norm
        dropout: dropout after layers
    """
    def __init__(self, in_dim: int, hidden_dims: list = [256, 256], out_dim: int = 128, use_bn: bool = True, dropout: float = 0.2):
        super().__init__()
        if SAGEConv is None:
            raise RuntimeError('torch_geometric not available, install torch-geometric to use GNNEncoder')
        dims = [in_dim] + hidden_dims
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        for i in range(len(dims)-1):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            if use_bn:
                self.bns.append(BatchNorm(dims[i+1]))
        self.project = nn.Sequential(
            nn.Linear(dims[-1], out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = torch.relu(h)
            if self.bns is not None:
                h = self.bns[i](h)
        z = self.project(h)
        return z


class CompatibilityScorer(nn.Module):
    """
    Scoring head that takes two node embeddings and returns a scalar compatibility score.
    Design: symmetric scoring via elementwise product + concat.
    """
    def __init__(self, emb_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # z_a, z_b: [batch, emb_dim]
        x = torch.cat([z_a, z_b], dim=-1)
        return self.mlp(x).squeeze(-1)


class OutfitGNNModel(nn.Module):
    """
    Combined model: encoder + scorer. Exposes convenient methods for encoding and scoring.
    """
    def __init__(self, in_dim: int, hidden_dims: list = [256,256], out_dim: int = 128, use_bn: bool = True, dropout: float = 0.2):
        super().__init__()
        self.encoder = GNNEncoder(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=out_dim, use_bn=use_bn, dropout=dropout)
        self.scorer = CompatibilityScorer(emb_dim=out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings."""
        return self.encoder(x, edge_index)

    def score_pairs(self, z: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        """Given a node embedding matrix `z`, score batches of pairs by index."""
        za = z[idx_a]
        zb = z[idx_b]
        return self.scorer(za, zb)
