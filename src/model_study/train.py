"""
train.py

Responsibilities:
 - Dataset wrappers for pairwise and full-graph training
 - Training loop with configurable objectives (margin-ranking / bce / contrastive)
 - Checkpointing and simple metrics (loss, recall@K)

Design:
 - Keep data loading separate from model training (Dataset classes only return indices)
 - Allow easily switching loss functions with a simple string in config

Note: This script is written to be run from a driver that provides prepared tensors: node_features (torch.Tensor), edge_index (torch.LongTensor), outfits (list of lists mapping to global node indices)

"""
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import os


class PairwiseDataset(Dataset):
    """
    Returns positive pairs (a,b) from outfits and sampled negatives (a, neg).
    Items are indices into the global node list.
    """
    def __init__(self, outfits: List[List[int]], num_nodes: int, negs_per_pos: int = 1, seed: int = 42):
        self.pos_pairs = []
        for outfit in outfits:
            L = len(outfit)
            for i in range(L):
                for j in range(i+1, L):
                    self.pos_pairs.append((int(outfit[i]), int(outfit[j])))
        self.num_nodes = num_nodes
        self.negs_per_pos = negs_per_pos
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        a,b = self.pos_pairs[idx]
        negs = []
        for _ in range(self.negs_per_pos):
            while True:
                n = int(self.rng.randint(0, self.num_nodes))
                if n != a and n != b:
                    break
            negs.append(n)
        return a, b, negs[0]


def recall_at_k(z: torch.Tensor, pos_pairs: List[Tuple[int,int]], k: int = 10) -> float:
    """
    Simple Recall@K: for each (a,b) check if b is in top-K nearest neighbors of a in embedding space.
    This is O(N^2) naive — suitable for small test sets.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    emb = z.cpu().numpy()
    sim = cosine_similarity(emb)
    hits = 0
    for a,b in pos_pairs:
        topk = np.argsort(-sim[a])[:k]
        if b in topk:
            hits += 1
    return hits / max(1, len(pos_pairs))


def train_loop(
    model: torch.nn.Module,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    outfits: List[List[int]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
    loss_type: str = 'margin',
    margin: float = 0.2,
    checkpoint_dir: str = './checkpoints'
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    dataset = PairwiseDataset(outfits, num_nodes=node_features.shape[0])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if loss_type == 'margin':
        loss_fn = nn.MarginRankingLoss(margin=margin)
    elif loss_type == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError('Unsupported loss_type')

    history = {'loss': [], 'recall@10': []}

    model.to(device)
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for a_idx, b_idx, neg_idx in loader:
            a_idx = a_idx.long().to(device)
            b_idx = b_idx.long().to(device)
            neg_idx = neg_idx.long().to(device)

            optimizer.zero_grad()
            z = model(node_features, edge_index)  # [N, emb]
            pos_scores = model.score_pairs(z, a_idx, b_idx)
            neg_scores = model.score_pairs(z, a_idx, neg_idx)

            if loss_type == 'margin':
                target = torch.ones_like(pos_scores).to(device)
                loss = loss_fn(pos_scores, neg_scores, target)
            elif loss_type == 'bce':
                # label positives as 1, negatives as 0, concatenate
                scores = torch.cat([pos_scores, neg_scores], dim=0)
                labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0).to(device)
                loss = loss_fn(scores, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * a_idx.size(0)

        avg_loss = total_loss / len(dataset)
        # compute simple recall@10 on a subset for speed
        model.eval()
        with torch.no_grad():
            z = model(node_features, edge_index)
            # sample up to 200 positive pairs
            sample_pairs = dataset.pos_pairs[:200]
            r10 = recall_at_k(z, sample_pairs, k=10)
        history['loss'].append(avg_loss)
        history['recall@10'].append(r10)
        print(f"Epoch {epoch}/{epochs} loss={avg_loss:.4f} recall@10={r10:.4f}")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt'))

    return model, history


# Example driver function that ties everything together (for script usage)
def run_training_example(node_features, edge_index, outfits, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = node_features.shape[1]
    model = None
    from src.models.outfit_gnn import OutfitGNNModel
    model = OutfitGNNModel(in_dim=in_dim, hidden_dims=[256,256], out_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model, history = train_loop(model, node_features, edge_index, outfits, optimizer, device, epochs=5, batch_size=128)
    return model, history