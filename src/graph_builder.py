from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
from collections import defaultdict


def build_knn_edges(embs: np.ndarray, k: int = 10, include_self: bool = False) -> List[Tuple[int,int,float]]:
    """
    Build directed k-NN edges using Euclidean distance.

    Args:
        embs: [N, D] numpy array of node embeddings
        k: number of neighbors
        include_self: whether to include self-loops

    Returns:
        List of (src_idx, dst_idx, weight) where weight is similarity (1/(1+dist)).
    """
    from sklearn.neighbors import NearestNeighbors
    N = embs.shape[0]
    if k >= N:
        k = N - 1
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embs)
    dists, inds = nbrs.kneighbors(embs)

    edges = []
    for i in range(N):
        for j_idx, dist in zip(inds[i], dists[i]):
            if not include_self and j_idx == i:
                continue
            if j_idx == i and include_self:
                sim = 1.0
            else:
                sim = 1.0 / (1.0 + float(dist))
            edges.append((int(i), int(j_idx), float(sim)))
    return edges


def build_cooccurrence_edges(outfits: List[List[int]]) -> List[Tuple[int,int,float]]:
    """
    Build undirected weighted edges from co-occurrence in outfits.
    Each unordered pair that appears together increments a cooccurrence count.

    Args:
        outfits: list of outfits, each a list of global node indices

    Returns:
        List of (src, dst, weight) representing co-occurrence counts (normalized)
    """
    co = defaultdict(int)
    for outfit in outfits:
        unique_items = list(dict.fromkeys(outfit))  # preserve order, drop duplicates
        L = len(unique_items)
        for i in range(L):
            for j in range(i+1, L):
                a = unique_items[i]
                b = unique_items[j]
                if a == b:
                    continue
                co[(a,b)] += 1
                co[(b,a)] += 1

    # normalize weights to [0,1]
    if not co:
        return []
    max_cnt = max(co.values())
    edges = [(int(a), int(b), float(cnt / max_cnt)) for (a,b), cnt in co.items()]
    return edges


def build_attribute_edges(node_attrs: List[Dict[str,Any]], attr_key: str = 'category') -> List[Tuple[int,int,float]]:
    """
    Build edges between nodes that share the same attribute value (e.g., same category or color).
    Weight is currently binary (1.0) or can be scaled by frequency.

    Args:
        node_attrs: list of dicts of attributes for each node index
        attr_key: key to use for matching

    Returns:
        List of edges (i, j, 1.0) for shared attribute
    """
    buckets = defaultdict(list)
    for idx, attrs in enumerate(node_attrs):
        val = attrs.get(attr_key, None)
        if val is None:
            continue
        buckets[val].append(idx)

    edges = []
    for val, indices in buckets.items():
        for i in indices:
            for j in indices:
                if i == j:
                    continue
                edges.append((int(i), int(j), 1.0))
    return edges


# --- Utilities: assemble and export ---

def normalize_edge_list(edges: List[Tuple[int,int,float]]) -> List[Tuple[int,int,float]]:
    """Merge edges with same (src,dst) by summing weights, then normalize to [0,1]."""
    acc = defaultdict(float)
    for a,b,w in edges:
        acc[(a,b)] += float(w)
    if not acc:
        return []
    max_w = max(acc.values())
    normalized = [(a,b, acc[(a,b)]/max_w) for (a,b) in acc.keys()]
    return normalized


def edges_to_edge_index_attr(edges: List[Tuple[int,int,float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert edges to PyG-style edge_index (2, E) and edge_attr (E, 1).
    """
    if not edges:
        return np.zeros((2,0), dtype=int), np.zeros((0,1), dtype=float)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    w = [e[2] for e in edges]
    edge_index = np.vstack([np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)])
    edge_attr = np.array(w, dtype=np.float32).reshape(-1,1)
    return edge_index, edge_attr


def assemble_graph(
    num_nodes: int,
    emb_matrix: Optional[np.ndarray] = None,
    outfits: Optional[List[List[int]]] = None,
    node_attrs: Optional[List[Dict[str,Any]]] = None,
    strategies: Optional[List[str]] = None,
    knn_k: int = 10
) -> Dict[str, Any]:
    """
    Assemble a graph according to requested strategies.

    Args:
        num_nodes: number of nodes
        emb_matrix: numpy array of shape [num_nodes, D]
        outfits: list of outfits (for cooccurrence)
        node_attrs: list of dicts, length num_nodes
        strategies: list of strategy names to apply: 'knn', 'cooccurrence', 'attr:KEY'
        knn_k: k for knn

    Returns:
        dict with keys: 'edge_list' (normalized), 'edge_index', 'edge_attr'
    """
    if strategies is None:
        strategies = ['knn', 'cooccurrence']

    all_edges = []
    for s in strategies:
        if s == 'knn':
            assert emb_matrix is not None, 'emb_matrix required for knn'
            e = build_knn_edges(emb_matrix, k=knn_k)
            all_edges.extend(e)
        elif s == 'cooccurrence':
            assert outfits is not None, 'outfits required for cooccurrence'
            e = build_cooccurrence_edges(outfits)
            all_edges.extend(e)
        elif s.startswith('attr:'):
            key = s.split(':',1)[1]
            assert node_attrs is not None, 'node_attrs required for attr edges'
            e = build_attribute_edges(node_attrs, attr_key=key)
            all_edges.extend(e)
        else:
            raise ValueError(f'Unknown strategy: {s}')

    norm_edges = normalize_edge_list(all_edges)
    edge_index, edge_attr = edges_to_edge_index_attr(norm_edges)

    return {
        'edge_list': norm_edges,
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }

