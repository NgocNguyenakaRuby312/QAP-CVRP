"""
utils/clustering.py
====================
K-Means clustering for large instances (N ≥ 100).                      # Phase 5

Decomposes one CVRP instance into K smaller sub-problems.
After clustering, kNN must be recomputed per sub-problem.              # P10
"""

import torch
from typing import List, Dict


def cluster_instance(
    coords: torch.Tensor,
    demands: torch.Tensor,
    n_clusters: int,
) -> List[Dict[str, torch.Tensor]]:
    """
    Cluster a single CVRP instance into K sub-problems using K-Means
    on customer coordinates (depot excluded from clustering).

    Args:
        coords:     [N+1, 2]  depot at index 0, customers at 1..N
        demands:    [N+1]     depot demand = 0
        n_clusters: K         number of clusters

    Returns:
        sub_problems: list of K dicts, each with keys:
            coords:  [Nk+1, 2]  depot (index 0) + cluster customers
            demands: [Nk+1]     depot demand + cluster demands
            indices: [Nk]       original customer indices (1-based)

    Note:
        After clustering, kNN must be recomputed per sub-problem (P10).
    """
    from sklearn.cluster import KMeans

    device = coords.device
    customer_coords = coords[1:].cpu().numpy()                         # [N, 2]

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(customer_coords)                           # [N]

    depot_coord = coords[0:1]                                          # [1, 2]
    depot_demand = demands[0:1]                                        # [1]

    sub_problems = []
    for k in range(n_clusters):
        # Customer indices in this cluster (0-based in customer array → 1-based in full)
        cust_mask = (labels == k)
        cust_indices = torch.where(torch.tensor(cust_mask))[0] + 1     # 1-based

        # Build sub-problem: depot + cluster customers
        sub_coords = torch.cat([
            depot_coord,
            coords[cust_indices],
        ], dim=0).to(device)                                           # [Nk+1, 2]

        sub_demands = torch.cat([
            depot_demand,
            demands[cust_indices],
        ], dim=0).to(device)                                           # [Nk+1]

        sub_problems.append({
            "coords":  sub_coords,
            "demands": sub_demands,
            "indices": cust_indices.to(device),                        # original 1-based indices
        })

    return sub_problems


def cluster_batch(
    coords: torch.Tensor,
    demands: torch.Tensor,
    n_clusters: int,
) -> List[List[Dict[str, torch.Tensor]]]:
    """
    Cluster a batch of instances.

    Args:
        coords:     [B, N+1, 2]
        demands:    [B, N+1]
        n_clusters: K

    Returns:
        List of B lists, each containing K sub-problem dicts.
    """
    B = coords.size(0)
    return [
        cluster_instance(coords[b], demands[b], n_clusters)
        for b in range(B)
    ]
