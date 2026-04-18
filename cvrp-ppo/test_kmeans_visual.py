"""
test_kmeans_visual.py
======================
Standalone diagnostic: verify utils/clustering.py K-Means works correctly.

Generates 200 random 2D points, runs cluster_instance(), and plots:
  - Points coloured by cluster
  - Centroid of each cluster marked with a star (computed from cluster means)
  - Depot marked separately (not clustered)

Run from cvrp-ppo/:
    python test_kmeans_visual.py

Output: outputs/kmeans_test.png
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils.clustering import cluster_instance
from utils.seed import set_seed

# ── Config ──────────────────────────────────────────────────────────────────
N_POINTS   = 200          # total customers (excluding depot)
N_CLUSTERS = 6            # K clusters
SEED       = 42
CAPACITY   = 40           # dummy capacity (clustering is geometry-only)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
SAVE_PATH  = os.path.join(OUTPUT_DIR, "kmeans_test.png")

COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # teal
]

# ── Generate 200 random points ───────────────────────────────────────────────
set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Depot at index 0, customers at 1..N_POINTS
depot    = torch.FloatTensor([[0.5, 0.5]])             # depot at centre
customers = torch.FloatTensor(N_POINTS, 2).uniform_(0, 1)

coords  = torch.cat([depot, customers], dim=0)         # [N+1, 2]
demands = torch.zeros(N_POINTS + 1, dtype=torch.long)
demands[1:] = torch.randint(1, 10, (N_POINTS,))       # demand ∈ [1,9]

print(f"Clustering {N_POINTS} points into K={N_CLUSTERS} clusters...")

# ── Run cluster_instance() ───────────────────────────────────────────────────
sub_problems = cluster_instance(coords, demands, N_CLUSTERS)

# ── Verify output structure ──────────────────────────────────────────────────
total_customers_assigned = sum(len(sp["indices"]) for sp in sub_problems)
print(f"\nCluster sizes:")
for k, sp in enumerate(sub_problems):
    n_k = len(sp["indices"])
    sum_demand = sp["demands"][1:].sum().item()
    centroid = sp["coords"][1:].mean(dim=0).numpy()   # mean of cluster (excl. depot)
    print(f"  Cluster {k+1}: {n_k:3d} nodes | demand sum = {sum_demand:4d} | "
          f"centroid = ({centroid[0]:.3f}, {centroid[1]:.3f})")

print(f"\nTotal customers assigned: {total_customers_assigned} / {N_POINTS}")
assert total_customers_assigned == N_POINTS, \
    f"FAIL: {total_customers_assigned} ≠ {N_POINTS}  (customers lost or duplicated)"
print("PASS: all 200 customers assigned exactly once\n")

# ── Check for duplicates ─────────────────────────────────────────────────────
all_indices = []
for sp in sub_problems:
    all_indices.extend(sp["indices"].tolist())
assert len(all_indices) == len(set(all_indices)), \
    f"FAIL: duplicate indices found in clusters"
print("PASS: no duplicate customer assignments\n")

# ── Plot ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor("white")

coords_np = coords.numpy()

# Depot
ax.plot(coords_np[0, 0], coords_np[0, 1], "r*",
        markersize=22, zorder=10, label="Depot", markeredgecolor="darkred")

# Clusters: scatter points + centroid star
for k, sp in enumerate(sub_problems):
    color  = COLORS[k % len(COLORS)]
    n_k    = len(sp["indices"])

    # Customer points (skip depot at index 0 of sub_coords)
    sub_coords_np = sp["coords"][1:].numpy()             # [Nk, 2]
    ax.scatter(sub_coords_np[:, 0], sub_coords_np[:, 1],
               color=color, s=40, alpha=0.8, zorder=4,
               label=f"Cluster {k+1}  ({n_k} nodes)")

    # Centroid = mean of cluster customer coords
    centroid = sub_coords_np.mean(axis=0)
    ax.plot(centroid[0], centroid[1], "*",
            color=color, markersize=18, zorder=8,
            markeredgecolor="black", markeredgewidth=0.8)
    ax.annotate(f"C{k+1}", xy=(centroid[0], centroid[1]),
                xytext=(4, 5), textcoords="offset points",
                fontsize=8, fontweight="bold", color="black")

# Centroid legend entry
ax.plot([], [], "*", color="grey", markersize=14,
        markeredgecolor="black", markeredgewidth=0.8, label="centroids")

ax.set_title(
    f"K-Means Clustering Diagnostic\n"
    f"N = {N_POINTS} random customers,  K = {N_CLUSTERS} clusters  "
    f"(sklearn KMeans, n_init=10)",
    fontsize=13,
)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_aspect("equal")

fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved: {SAVE_PATH}")

# ── Extra: test with DIFFERENT K values ──────────────────────────────────────
print("\nTesting robustness across K values:")
for k_test in [3, 5, 8, 10]:
    sps = cluster_instance(coords, demands, k_test)
    total = sum(len(sp["indices"]) for sp in sps)
    sizes = [len(sp["indices"]) for sp in sps]
    ok = "PASS" if total == N_POINTS else "FAIL"
    print(f"  K={k_test:2d}: {ok}  sizes={sizes}  sum={total}")
