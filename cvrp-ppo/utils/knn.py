"""
utils/knn.py
============
Spatial k-nearest neighbours — precomputed once per instance, reused every decode step.

    knn_indices: [B, N+1, k]   k=5, no self-loops                     # §3, §10
"""

import torch


def compute_knn(coords: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Compute k-nearest neighbours on spatial coordinates.

    Args:
        coords: [B, N+1, 2]  node coordinates (depot + customers)
        k:      number of neighbours (default 5)

    Returns:
        knn_indices: [B, N+1, k]  on same device as coords, no self-loops
    """
    k = min(k, coords.size(1) - 1)
    dists = torch.cdist(coords, coords)                                # [B, N+1, N+1]
    dists.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))               # P4: no self-loops
    _, knn_idx = dists.topk(k, dim=-1, largest=False)                  # [B, N+1, k]
    return knn_idx                                                     # inherits device from coords
