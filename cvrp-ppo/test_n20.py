#!/usr/bin/env python
"""test_n20.py — Evaluate trained QAP-DRL on CVRP-20 test set."""

import os, sys, time, json
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

from models.qap_policy import QAPPolicy
from environment.cvrp_env import CVRPEnv
from utils.data_generator import load_dataset

# ═══════════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════════
GRAPH_SIZE      = 20
CAPACITY        = 30
LKH3_REF        = 6.10
TEST_PATH       = os.path.join(SCRIPT_DIR, "datasets", "test_n20.pkl")
MODEL_PATH      = os.path.join(SCRIPT_DIR, "outputs", "n20", "best_model.pt")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "outputs", "n20")
EVAL_BATCH_SIZE = 256


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def check_feasibility(demands, capacity, actions):
    """Vectorised feasibility check. Returns number of feasible routes."""
    B, T = actions.shape
    used = torch.zeros(B, device=actions.device)
    violated = torch.zeros(B, dtype=torch.bool, device=actions.device)
    for t in range(T):
        at_depot = (actions[:, t] == 0)
        used = torch.where(at_depot, torch.zeros_like(used), used)
        d = demands.gather(1, actions[:, t:t + 1]).squeeze(1)
        used = used + d
        violated = violated | (used > capacity + 1e-6)
    return (~violated).sum().item()


def plot_route_map(coords_np, actions_list, graph_size, tour_length,
                   title, save_path):
    """Plot a single CVRP solution with coloured vehicle routes and arrows."""
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    routes = []
    cur = []
    for node in actions_list:
        if node == 0:
            if cur:
                routes.append(cur)
            cur = []
        else:
            cur.append(node)
    if cur:
        routes.append(cur)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")

    ax.plot(coords_np[0, 0], coords_np[0, 1], "r*",
            markersize=20, label="Depot", zorder=5)

    for i in range(1, graph_size + 1):
        ax.plot(coords_np[i, 0], coords_np[i, 1], "o",
                color="#4a90d9", markersize=6, zorder=4)
        ax.annotate(str(i), (coords_np[i, 0], coords_np[i, 1]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)

    for r_idx, route in enumerate(routes):
        color = COLORS[r_idx % len(COLORS)]
        path = [0] + route + [0]
        for k in range(len(path) - 1):
            x0, y0 = coords_np[path[k]]
            x1, y1 = coords_np[path[k + 1]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        ax.plot([], [], color=color, lw=1.5, label=f"Route {r_idx + 1}")

    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_aspect("equal")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("       Train first with: python train_n20.py")
        sys.exit(1)

    policy = QAPPolicy()
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy.to(device)
    policy.eval()

    n_params = sum(p.numel() for p in policy.parameters())

    # ── Load test data ──────────────────────────────────────────────────
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: Test dataset not found at {TEST_PATH}")
        sys.exit(1)

    coords, demands, capacity = load_dataset(TEST_PATH, device=str(device))
    B_total = coords.size(0)

    print("=" * 60)
    print(f"  QAP-DRL Test — CVRP-{GRAPH_SIZE}")
    print("=" * 60)
    print(f"  Model       : {MODEL_PATH}")
    print(f"  Test set    : {TEST_PATH}  ({B_total} instances)")
    print(f"  Device      : {device}")
    print(f"  Parameters  : {n_params}")
    print("=" * 60)

    # ── Greedy evaluation in batches ────────────────────────────────────
    all_dists = []
    n_feasible = 0
    best_inst_dist = float("inf")
    best_inst_coords = None
    best_inst_actions = None
    t_start = time.time()

    n_batches = (B_total + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
    for i in tqdm(range(0, B_total, EVAL_BATCH_SIZE),
                  total=n_batches, desc="Testing", unit="batch"):
        j = min(i + EVAL_BATCH_SIZE, B_total)
        bc = coords[i:j]
        bd = demands[i:j]
        B = bc.size(0)

        env = CVRPEnv(num_loc=GRAPH_SIZE, device=str(device))
        state = env.reset({
            "coords": bc,
            "demands": bd,
            "capacity": torch.full((B,), float(capacity), device=device),
        })

        with torch.no_grad():
            actions, _, _ = policy(state, env, deterministic=True)

        # Tour distances
        T = actions.shape[1]
        idx = actions.unsqueeze(-1).expand(B, T, 2)
        route = bc.gather(1, idx)
        depot = bc[:, 0:1, :]
        full = torch.cat([depot, route, depot], dim=1)
        dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)
        all_dists.append(dists)

        # Track best instance for route map
        batch_best = dists.argmin().item()
        if dists[batch_best].item() < best_inst_dist:
            best_inst_dist = dists[batch_best].item()
            best_inst_coords = bc[batch_best].cpu()
            best_inst_actions = actions[batch_best].cpu()

        # Feasibility
        n_feasible += check_feasibility(bd, capacity, actions)

    total_time = time.time() - t_start
    all_dists = torch.cat(all_dists)

    # ── Statistics ──────────────────────────────────────────────────────
    avg = all_dists.mean().item()
    std = all_dists.std().item()
    mn = all_dists.min().item()
    mx = all_dists.max().item()
    gap = 100.0 * (avg - LKH3_REF) / LKH3_REF
    feas_rate = n_feasible / B_total
    ms_per_inst = (total_time / B_total) * 1000

    print()
    print("=" * 60)
    print(f"  Results — CVRP-{GRAPH_SIZE}  ({B_total} instances)")
    print("=" * 60)
    print(f"  Avg tour length    : {avg:.4f}")
    print(f"  vs LKH3 gap        : {gap:+.2f}%  (LKH3 ref = {LKH3_REF})")
    print(f"  Min tour length    : {mn:.4f}")
    print(f"  Max tour length    : {mx:.4f}")
    print(f"  Std tour length    : {std:.4f}")
    print(f"  Feasibility rate   : {feas_rate * 100:.2f}%"
          f"  ({n_feasible}/{B_total})")
    print(f"  Inference time     : {ms_per_inst:.2f} ms/instance")
    print(f"  Total time         : {total_time:.1f}s")
    print("=" * 60)

    # ── Save results ────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "graph_size": GRAPH_SIZE,
        "n_instances": B_total,
        "avg_tour_length": avg,
        "lkh3_ref": LKH3_REF,
        "gap_pct": gap,
        "min_tour": mn,
        "max_tour": mx,
        "std_tour": std,
        "feasibility_rate": feas_rate,
        "ms_per_instance": ms_per_inst,
        "total_time_s": total_time,
        "model_path": MODEL_PATH,
    }
    out_path = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Tour length distribution histogram ──────────────────────────────
    dists_np = all_dists.cpu().numpy()

    fig_hist, ax_h = plt.subplots(figsize=(8, 5))
    fig_hist.patch.set_facecolor("white")
    ax_h.hist(dists_np, bins=50, color="#1f77b4", edgecolor="white",
              alpha=0.85, label="Tour lengths")
    ax_h.axvline(avg, color="red", linestyle="--", linewidth=1.5,
                 label=f"Mean = {avg:.2f}")
    ax_h.axvline(LKH3_REF, color="green", linestyle="--", linewidth=1.5,
                 label=f"LKH3 = {LKH3_REF}")
    ax_h.set_title(f"Tour Length Distribution — CVRP-{GRAPH_SIZE} Test Set",
                   fontsize=14)
    ax_h.set_xlabel("Tour Length", fontsize=12)
    ax_h.set_ylabel("Count", fontsize=12)
    ax_h.annotate(f"Mean: {avg:.2f}\nGap: {gap:+.1f}%",
                  xy=(0.97, 0.95), xycoords="axes fraction",
                  ha="right", va="top", fontsize=11,
                  bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))
    ax_h.legend(fontsize=10)

    hist_path = os.path.join(OUTPUT_DIR, "test_distribution.png")
    fig_hist.savefig(hist_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {hist_path}")

    # ── Best test route map ─────────────────────────────────────────────
    plot_route_map(
        best_inst_coords.numpy(),
        best_inst_actions.tolist(),
        GRAPH_SIZE, best_inst_dist,
        f"Best Test Route — CVRP-{GRAPH_SIZE} | Tour Length: {best_inst_dist:.2f}",
        os.path.join(OUTPUT_DIR, "test_best_route.png"),
    )

    plt.show()


if __name__ == "__main__":
    main()
