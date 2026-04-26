#!/usr/bin/env python
"""
test_n200.py — Evaluate trained QAP-DRL on CVRP-200 test set.
Run after train_n200.py completes:  python test_n200.py
"""

import os, sys, time, json
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
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
from utils.ortools_solver import ORTOOLS_OK

# ═══════════════════════════════════════════════════════════════════════════
# Settings — must match train_n200.py
# ═══════════════════════════════════════════════════════════════════════════
GRAPH_SIZE      = 200
CAPACITY        = 50
AMP_DIM         = 4
HIDDEN_DIM      = 32
KNN_K           = 30

TEST_PATH       = os.path.join(SCRIPT_DIR, "datasets", "test_n200.pkl")
MODEL_PATH      = os.path.join(SCRIPT_DIR, "outputs", "n200", "best_model.pt")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "outputs", "n200")

EVAL_BATCH_SIZE = 64               # N=200 uses significant VRAM
AUG_SAMPLES     = 8
ORTOOLS_TEST_N  = 50               # N=200 is very slow for OR-Tools
ORTOOLS_TIME    = 10.0             # 10s per instance


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def check_feasibility(demands, capacity, actions):
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


def plot_route_map(coords_np, actions_list, graph_size, tour_length, title, save_path):
    COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    routes, cur = [], []
    for node in actions_list:
        if node == 0:
            if cur: routes.append(cur); cur = []
        else: cur.append(node)
    if cur: routes.append(cur)
    fig, ax = plt.subplots(figsize=(12, 12)); fig.patch.set_facecolor("white")
    ax.plot(coords_np[0,0], coords_np[0,1], "r*", markersize=20, label="Depot", zorder=5)
    for i in range(1, graph_size+1):
        ax.plot(coords_np[i,0], coords_np[i,1], "o", color="#4a90d9", markersize=3, zorder=4)
    for r_idx, route in enumerate(routes):
        color = COLORS[r_idx % len(COLORS)]; path = [0]+route+[0]
        for k in range(len(path)-1):
            x0,y0 = coords_np[path[k]]; x1,y1 = coords_np[path[k+1]]
            ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {save_path}")


def run_ortools_on_test(coords, demands, capacity, n_instances, time_limit):
    if not ORTOOLS_OK:
        print("  OR-Tools not available, skipping.")
        return None
    from utils.ortools_solver import solve_one_with_routes
    results = []
    print(f"\n  Running OR-Tools on {n_instances} test instances (time_limit={time_limit}s)...")
    for i in tqdm(range(n_instances), desc="  OR-Tools", unit="inst"):
        c = coords[i].cpu().numpy()
        d = demands[i].cpu().numpy().astype(int)
        cost, _ = solve_one_with_routes(c, d, capacity, time_limit)
        if cost is not None: results.append(cost)
    if results:
        avg = sum(results) / len(results)
        print(f"  OR-Tools avg on test: {avg:.4f}  ({len(results)}/{n_instances} solved)")
        return avg
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("       Train first with: python train_n200.py")
        sys.exit(1)

    policy = QAPPolicy(knn_k=KNN_K, amp_dim=AMP_DIM, hidden_dim=HIDDEN_DIM)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt)
    policy.to(device).eval()
    n_params = sum(p.numel() for p in policy.parameters())

    if not os.path.exists(TEST_PATH):
        print(f"ERROR: Test dataset not found at {TEST_PATH}")
        print("       Run: python generate_n200_datasets.py")
        sys.exit(1)

    coords, demands, capacity = load_dataset(TEST_PATH, device=str(device))
    B_total = coords.size(0)

    # OR-Tools reference
    print(f"\n  [0/3] Computing OR-Tools reference ({ORTOOLS_TEST_N} instances)...")
    ORTOOLS_REF = run_ortools_on_test(coords, demands, capacity, ORTOOLS_TEST_N, ORTOOLS_TIME)
    if ORTOOLS_REF is None:
        ORTOOLS_REF = 1.0  # fallback
        print("  WARNING: OR-Tools failed, using placeholder ref=1.0")

    print("=" * 68)
    print(f"  QAP-DRL Test — CVRP-{GRAPH_SIZE} ({AMP_DIM}D, {n_params} params)")
    print("=" * 68)

    # Greedy evaluation
    print("\n  [1/3] Greedy evaluation...")
    all_greedy_dists = []
    n_feasible = 0
    best_dist = float("inf"); worst_dist = float("-inf")
    best_coords = best_actions = worst_coords = worst_actions = None
    t_start = time.time()

    n_batches = (B_total + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
    for i in tqdm(range(0, B_total, EVAL_BATCH_SIZE), total=n_batches, desc="  Greedy"):
        j = min(i + EVAL_BATCH_SIZE, B_total)
        bc, bd, B = coords[i:j], demands[i:j], min(EVAL_BATCH_SIZE, j-i)
        env = CVRPEnv(num_loc=GRAPH_SIZE, device=str(device))
        state = env.reset({"coords": bc, "demands": bd,
                           "capacity": torch.full((B,), float(capacity), device=device)})
        with torch.no_grad():
            actions, _, _ = policy(state, env, deterministic=True)
        T = actions.shape[1]
        idx = actions.unsqueeze(-1).expand(B, T, 2)
        route = bc.gather(1, idx); depot = bc[:, 0:1, :]
        full = torch.cat([depot, route, depot], dim=1)
        dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)
        all_greedy_dists.append(dists)
        bi = dists.argmin().item(); wi = dists.argmax().item()
        if dists[bi].item() < best_dist:
            best_dist = dists[bi].item(); best_coords = bc[bi].cpu(); best_actions = actions[bi].cpu()
        if dists[wi].item() > worst_dist:
            worst_dist = dists[wi].item(); worst_coords = bc[wi].cpu(); worst_actions = actions[wi].cpu()
        n_feasible += check_feasibility(bd, capacity, actions)

    greedy_time = time.time() - t_start
    all_greedy_dists = torch.cat(all_greedy_dists)
    greedy_avg = all_greedy_dists.mean().item()

    # Augmented evaluation
    print(f"\n  [2/3] Augmented evaluation (x{AUG_SAMPLES})...")
    from training.evaluate import _aug_transforms
    transforms = _aug_transforms()
    all_aug_dists = []
    t_start_aug = time.time()

    for i in tqdm(range(0, B_total, EVAL_BATCH_SIZE), total=n_batches, desc="  Aug"):
        j = min(i + EVAL_BATCH_SIZE, B_total)
        bc, bd, B = coords[i:j].to(device), demands[i:j].to(device), min(EVAL_BATCH_SIZE, j-i)
        best_batch = None
        for t_idx in range(min(AUG_SAMPLES, 8)):
            aug_c = transforms[t_idx](bc)
            env = CVRPEnv(num_loc=GRAPH_SIZE, device=str(device))
            state = env.reset({"coords": aug_c, "demands": bd,
                               "capacity": torch.full((B,), float(capacity), device=device)})
            with torch.no_grad():
                actions, _, _ = policy(state, env, deterministic=True)
            T = actions.shape[1]
            idx = actions.unsqueeze(-1).expand(B, T, 2)
            route = bc.gather(1, idx); depot = bc[:, 0:1, :]
            full = torch.cat([depot, route, depot], dim=1)
            dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)
            best_batch = dists if best_batch is None else torch.minimum(best_batch, dists)
        all_aug_dists.append(best_batch)

    aug_time = time.time() - t_start_aug
    all_aug_dists = torch.cat(all_aug_dists)
    aug_avg = all_aug_dists.mean().item()
    aug_std = all_aug_dists.std().item()

    # Results
    feas_rate = n_feasible / B_total
    gap_ort = 100.0 * (aug_avg - ORTOOLS_REF) / ORTOOLS_REF
    greedy_gap = 100.0 * (greedy_avg - ORTOOLS_REF) / ORTOOLS_REF
    p10, p25, p50, p75, p90 = [torch.quantile(all_aug_dists, q).item() for q in [.1,.25,.5,.75,.9]]

    print()
    print("=" * 68)
    print(f"  TEST RESULTS — CVRP-{GRAPH_SIZE}  ({B_total} instances)")
    print("=" * 68)
    print(f"  Greedy (no aug) : {greedy_avg:.4f}  (gap: {greedy_gap:+.2f}%)")
    print(f"  Augmented (x{AUG_SAMPLES}) : {aug_avg:.4f}  (gap: {gap_ort:+.2f}%)")
    print(f"  OR-Tools ref    : {ORTOOLS_REF:.4f}")
    print(f"  Std             : {aug_std:.4f}")
    print(f"  Percentiles: p10={p10:.3f} p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} p90={p90:.3f}")
    print(f"  Feasibility     : {feas_rate*100:.2f}%")
    print(f"  Parameters      : {n_params}")
    print("=" * 68)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "graph_size": GRAPH_SIZE, "amp_dim": AMP_DIM, "hidden_dim": HIDDEN_DIM,
        "n_params": n_params, "n_instances": B_total,
        "greedy_avg": greedy_avg, "aug_avg": aug_avg, "aug_std": aug_std,
        "gap_ortools_pct": gap_ort, "ortools_ref": ORTOOLS_REF,
        "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
        "feasibility_rate": feas_rate,
    }
    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 5)); fig.patch.set_facecolor("white")
    ax.hist(all_aug_dists.cpu().numpy(), bins=50, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.axvline(aug_avg, color="red", ls="--", lw=1.5, label=f"Mean = {aug_avg:.3f}")
    ax.axvline(ORTOOLS_REF, color="green", ls="--", lw=1.5, label=f"ORT = {ORTOOLS_REF:.3f}")
    ax.set_title(f"Tour Length Distribution — CVRP-{GRAPH_SIZE} Test"); ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "test_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Route maps
    if best_coords is not None:
        plot_route_map(best_coords.numpy(), best_actions.tolist(), GRAPH_SIZE, best_dist,
                       f"Best Test Route — CVRP-{GRAPH_SIZE} | Tour: {best_dist:.3f}",
                       os.path.join(OUTPUT_DIR, "test_best_route.png"))
    if worst_coords is not None:
        plot_route_map(worst_coords.numpy(), worst_actions.tolist(), GRAPH_SIZE, worst_dist,
                       f"Worst Test Route — CVRP-{GRAPH_SIZE} | Tour: {worst_dist:.3f}",
                       os.path.join(OUTPUT_DIR, "test_worst_route.png"))
    print("\n  Done.")


if __name__ == "__main__":
    main()
