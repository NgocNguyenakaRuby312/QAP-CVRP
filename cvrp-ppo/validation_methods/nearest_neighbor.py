"""
validation_methods/nearest_neighbor.py
========================================
Nearest Neighbour greedy heuristic for CVRP.

No external dependencies — pure NumPy.

Usage:
    python nearest_neighbor.py --n 10
    python nearest_neighbor.py --n 20
    python nearest_neighbor.py --n 50
    python nearest_neighbor.py --n 100
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_generator import load_dataset

DATASETS = {
    10:  ("datasets/val_n10.pkl",  20),
    20:  ("datasets/val_n20.pkl",  30),
    50:  ("datasets/val_n50.pkl",  40),
    100: ("datasets/val_n100.pkl", 50),
}

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..")


def nn_cvrp(coords: np.ndarray, demands: np.ndarray, capacity: int) -> float:
    """
    Nearest-Neighbour CVRP heuristic for one instance.

    Args:
        coords:   [N+1, 2]  node coordinates (depot at index 0)
        demands:  [N+1]     integer demands  (depot demand = 0)
        capacity: int       vehicle capacity

    Returns:
        total_distance: float
    """
    N = len(demands) - 1
    visited = np.zeros(N + 1, dtype=bool)
    visited[0] = True          # depot is never "visited" as a customer

    current     = 0
    remaining   = capacity
    total_dist  = 0.0

    while not visited[1:].all():
        # Find nearest feasible unvisited customer
        best_j    = -1
        best_d    = np.inf
        for j in range(1, N + 1):
            if visited[j]:
                continue
            if demands[j] > remaining:
                continue
            d = np.linalg.norm(coords[current] - coords[j])
            if d < best_d:
                best_d = d
                best_j = j

        if best_j == -1:
            # No feasible customer — return to depot, reset capacity
            total_dist += np.linalg.norm(coords[current] - coords[0])
            current    = 0
            remaining  = capacity
        else:
            total_dist += best_d
            visited[best_j] = True
            remaining      -= demands[best_j]
            current         = best_j

    # Final return to depot
    total_dist += np.linalg.norm(coords[current] - coords[0])
    return total_dist


def evaluate_nn(n: int, n_instances: int = 1000) -> float:
    val_path, capacity = DATASETS[n]
    val_path = os.path.join(SCRIPT_DIR, val_path)

    coords_t, demands_t, _ = load_dataset(val_path, device="cpu")
    coords_np  = coords_t[:n_instances].numpy()    # [M, N+1, 2]
    demands_np = demands_t[:n_instances].numpy()   # [M, N+1]

    total = 0.0
    for i in range(len(coords_np)):
        total += nn_cvrp(coords_np[i], demands_np[i], capacity)

    avg = total / len(coords_np)
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=[10, 20, 50, 100], default=10)
    parser.add_argument("--instances", type=int, default=1000)
    args = parser.parse_args()

    print(f"Nearest Neighbour — CVRP-{args.n}")
    print(f"  Instances : {args.instances}")
    avg = evaluate_nn(args.n, args.instances)
    print(f"  Avg tour  : {avg:.4f}")
