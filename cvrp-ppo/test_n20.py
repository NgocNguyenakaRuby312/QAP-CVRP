#!/usr/bin/env python
"""
test_n20.py — Evaluate trained QAP-DRL on CVRP-20 test set.

Produces:
    - test_results.json        comprehensive stats
    - test_distribution.png    tour length histogram
    - test_best_route.png      best test route visualization
    - test_worst_route.png     worst test route (for analysis)

Uses augmented evaluation (8 coord transforms × greedy) for best results.
Compares against OR-Tools reference (computed on same test set).
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
from training.evaluate import evaluate_augmented
from utils.data_generator import load_dataset
from utils.ortools_solver import ORTOOLS_OK

# ═══════════════════════════════════════════════════════════════════════════
# Settings — must match train_n20.py
# ═══════════════════════════════════════════════════════════════════════════
GRAPH_SIZE      = 20
CAPACITY        = 30
AMP_DIM         = 4                # Phase 2: 4D amplitudes
HIDDEN_DIM      = 32               # Fix 4: rotation MLP hidden width
KNN_K           = 10

# References
ORTOOLS_REF     = 6.1509           # from training OR-Tools benchmark (1K instances)
# Paths
TEST_PATH       = os.path.join(SCRIPT_DIR, "datasets", "test_n20.pkl")
MODEL_PATH      = os.path.join(SCRIPT_DIR, "outputs", "n20", "best_model.pt")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "outputs", "n20")

# Evaluation
EVAL_BATCH_SIZE = 256
AUG_SAMPLES     = 8                # coordinate augmentation × greedy
ORTOOLS_TEST_N  = 100              # OR-Tools on first N test instances for comparison
ORTOOLS_TIME    = 2.0              # seconds per instance


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
    routes, cur = [], []
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
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_ortools_on_test(coords, demands, capacity, n_instances, time_limit):
    """Run OR-Tools on first n_instances of test set for comparison."""
    if not ORTOOLS_OK:
        print("  OR-Tools not available, skipping comparison.")
        return None

    from utils.ortools_solver import solve_one_with_routes
    results = []
    print(f"\n  Running OR-Tools on {n_instances} test instances (time_limit={time_limit}s)...")
    for i in tqdm(range(n_instances), desc="  OR-Tools", unit="inst"):
        c = coords[i].cpu().numpy()
        d = demands[i].cpu().numpy().astype(int)
        cost, _ = solve_one_with_routes(c, d, capacity, time_limit)
        if cost is not None:
            results.append(cost)
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

    # ── Load model ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("       Train first with: python train_n20.py")
        sys.exit(1)

    # Must match training config exactly
    policy = QAPPolicy(
        knn_k=KNN_K,
        amp_dim=AMP_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt)
    policy.to(device)
    policy.eval()

    n_params = sum(p.numel() for p in policy.parameters())

    # ── Load test data ──────────────────────────────────────────────────
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: Test dataset not found at {TEST_PATH}")
        sys.exit(1)

    coords, demands, capacity = load_dataset(TEST_PATH, device=str(device))
    B_total = coords.size(0)

    print("=" * 68)
    print(f"  QAP-DRL Test — CVRP-{GRAPH_SIZE} (Phase 2: {AMP_DIM}D amplitudes)")
    print("=" * 68)
    print(f"  Model       : {MODEL_PATH}")
    print(f"  Test set    : {TEST_PATH}  ({B_total} instances)")
    print(f"  Device      : {device}")
    print(f"  Parameters  : {n_params}")
    print(f"  Amp dim     : {AMP_DIM}D (S{AMP_DIM-1} hypersphere)")
    print(f"  Hidden dim  : {HIDDEN_DIM}")
    print(f"  Aug samples : {AUG_SAMPLES}")
    print("=" * 68)

    # ── Greedy evaluation (no augmentation) ─────────────────────────────
    print("\n  [1/3] Greedy evaluation (deterministic, no augmentation)...")
    all_greedy_dists = []
    n_feasible = 0
    best_dist = float("inf")
    worst_dist = float("-inf")
    best_coords = best_actions = None
    worst_coords = worst_actions = None
    t_start = time.time()

    n_batches = (B_total + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
    for i in tqdm(range(0, B_total, EVAL_BATCH_SIZE),
                  total=n_batches, desc="  Greedy", unit="batch"):
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

        T = actions.shape[1]
        idx = actions.unsqueeze(-1).expand(B, T, 2)
        route = bc.gather(1, idx)
        depot = bc[:, 0:1, :]
        full = torch.cat([depot, route, depot], dim=1)
        dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)
        all_greedy_dists.append(dists)

        batch_best = dists.argmin().item()
        batch_worst = dists.argmax().item()
        if dists[batch_best].item() < best_dist:
            best_dist = dists[batch_best].item()
            best_coords = bc[batch_best].cpu()
            best_actions = actions[batch_best].cpu()
        if dists[batch_worst].item() > worst_dist:
            worst_dist = dists[batch_worst].item()
            worst_coords = bc[batch_worst].cpu()
            worst_actions = actions[batch_worst].cpu()

        n_feasible += check_feasibility(bd, capacity, actions)

    greedy_time = time.time() - t_start
    all_greedy_dists = torch.cat(all_greedy_dists)
    greedy_avg = all_greedy_dists.mean().item()
    greedy_std = all_greedy_dists.std().item()

    # ── Augmented evaluation (8 transforms × greedy) ────────────────────
    print(f"\n  [2/3] Augmented evaluation ({AUG_SAMPLES} transforms × greedy)...")
    all_aug_dists = []
    t_start_aug = time.time()

    for i in tqdm(range(0, B_total, EVAL_BATCH_SIZE),
                  total=n_batches, desc="  Aug×8", unit="batch"):
        j = min(i + EVAL_BATCH_SIZE, B_total)
        bc = coords[i:j].to(device)
        bd = demands[i:j].to(device)

        aug_tour = evaluate_augmented(
            policy, (bc, bd, capacity), device, n_samples=AUG_SAMPLES
        )
        # evaluate_augmented returns mean — we need per-instance
        # Re-run manually for per-instance stats
        from training.evaluate import _aug_transforms
        transforms = _aug_transforms()
        B = bc.size(0)
        best_dists_batch = None

        for t_idx in range(min(AUG_SAMPLES, 8)):
            aug_coords = transforms[t_idx](bc)
            env = CVRPEnv(num_loc=GRAPH_SIZE, device=str(device))
            state = env.reset({
                "coords": aug_coords,
                "demands": bd,
                "capacity": torch.full((B,), float(capacity), device=device),
            })
            with torch.no_grad():
                actions, _, _ = policy(state, env, deterministic=True)

            T = actions.shape[1]
            idx = actions.unsqueeze(-1).expand(B, T, 2)
            route = bc.gather(1, idx)
            depot = bc[:, 0:1, :]
            full = torch.cat([depot, route, depot], dim=1)
            dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)

            if best_dists_batch is None:
                best_dists_batch = dists
            else:
                best_dists_batch = torch.minimum(best_dists_batch, dists)

        all_aug_dists.append(best_dists_batch)

    aug_time = time.time() - t_start_aug
    all_aug_dists = torch.cat(all_aug_dists)
    aug_avg = all_aug_dists.mean().item()
    aug_std = all_aug_dists.std().item()

    # ── OR-Tools comparison on test set ─────────────────────────────────
    print(f"\n  [3/3] OR-Tools comparison ({ORTOOLS_TEST_N} instances)...")
    ortools_test_avg = run_ortools_on_test(
        coords, demands, capacity, ORTOOLS_TEST_N, ORTOOLS_TIME
    )

    # ── Statistics ──────────────────────────────────────────────────────
    feas_rate = n_feasible / B_total
    gap_ortools = 100.0 * (aug_avg - ORTOOLS_REF) / ORTOOLS_REF
    greedy_gap = 100.0 * (greedy_avg - ORTOOLS_REF) / ORTOOLS_REF
    ms_greedy = (greedy_time / B_total) * 1000
    ms_aug = (aug_time / B_total) * 1000

    p10 = torch.quantile(all_aug_dists, 0.10).item()
    p25 = torch.quantile(all_aug_dists, 0.25).item()
    p50 = torch.quantile(all_aug_dists, 0.50).item()
    p75 = torch.quantile(all_aug_dists, 0.75).item()
    p90 = torch.quantile(all_aug_dists, 0.90).item()

    print()
    print("=" * 68)
    print(f"  TEST RESULTS — CVRP-{GRAPH_SIZE}  ({B_total} instances)")
    print("=" * 68)
    print(f"  Greedy tour (no aug)  : {greedy_avg:.4f}  (gap vs ORT: {greedy_gap:+.2f}%)")
    print(f"  Augmented tour (×{AUG_SAMPLES})  : {aug_avg:.4f}  (gap vs ORT: {gap_ortools:+.2f}%)")
    print(f"  OR-Tools ref (train) : {ORTOOLS_REF:.4f}")
    if ortools_test_avg:
        gap_test_ort = 100.0 * (aug_avg - ortools_test_avg) / ortools_test_avg
        print(f"  OR-Tools ref (test)  : {ortools_test_avg:.4f}  (gap: {gap_test_ort:+.2f}%)")
    print("  " + "-" * 64)
    print(f"  Std (augmented)      : {aug_std:.4f}")
    print(f"  Min tour             : {all_aug_dists.min().item():.4f}")
    print(f"  Max tour             : {all_aug_dists.max().item():.4f}")
    print(f"  Percentiles: p10={p10:.3f} p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} p90={p90:.3f}")
    print("  " + "-" * 64)
    print(f"  Feasibility          : {feas_rate * 100:.2f}% ({n_feasible}/{B_total})")
    print(f"  Greedy speed         : {ms_greedy:.2f} ms/instance")
    print(f"  Augmented speed      : {ms_aug:.2f} ms/instance")
    print(f"  Parameters           : {n_params}")
    print("=" * 68)

    # ── Save results JSON ───────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "graph_size": GRAPH_SIZE,
        "amp_dim": AMP_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_params": n_params,
        "n_instances": B_total,
        "greedy_avg": greedy_avg,
        "greedy_std": greedy_std,
        "greedy_gap_ortools_pct": greedy_gap,
        "aug_avg": aug_avg,
        "aug_std": aug_std,
        "aug_gap_ortools_pct": gap_ortools,
        "ortools_ref_train": ORTOOLS_REF,
        "ortools_ref_test": ortools_test_avg,
        "min_tour": all_aug_dists.min().item(),
        "max_tour": all_aug_dists.max().item(),
        "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
        "feasibility_rate": feas_rate,
        "ms_per_instance_greedy": ms_greedy,
        "ms_per_instance_aug": ms_aug,
        "aug_samples": AUG_SAMPLES,
        "model_path": MODEL_PATH,
    }
    out_path = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Tour length distribution histogram ──────────────────────────────
    dists_np = all_aug_dists.cpu().numpy()

    fig_hist, ax_h = plt.subplots(figsize=(10, 5))
    fig_hist.patch.set_facecolor("white")
    ax_h.hist(dists_np, bins=50, color="#1f77b4", edgecolor="white",
              alpha=0.85, label="Tour lengths (aug×8)")
    ax_h.axvline(aug_avg, color="red", linestyle="--", linewidth=1.5,
                 label=f"Mean = {aug_avg:.3f}")
    ax_h.axvline(ORTOOLS_REF, color="green", linestyle="--", linewidth=1.5,
                 label=f"OR-Tools = {ORTOOLS_REF:.3f}")
    ax_h.set_title(f"Tour Length Distribution — CVRP-{GRAPH_SIZE} Test Set "
                   f"({B_total} instances)", fontsize=14)
    ax_h.set_xlabel("Tour Length", fontsize=12)
    ax_h.set_ylabel("Count", fontsize=12)
    ax_h.annotate(
        f"Aug×{AUG_SAMPLES} Mean: {aug_avg:.3f}\n"
        f"Gap vs ORT: {gap_ortools:+.2f}%\n"
        f"Params: {n_params}",
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))
    ax_h.legend(fontsize=10)
    hist_path = os.path.join(OUTPUT_DIR, "test_distribution.png")
    fig_hist.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig_hist)
    print(f"  Saved: {hist_path}")

    # ── Best test route map ─────────────────────────────────────────────
    if best_coords is not None:
        plot_route_map(
            best_coords.numpy(), best_actions.tolist(),
            GRAPH_SIZE, best_dist,
            f"Best Test Route — CVRP-{GRAPH_SIZE} | Tour: {best_dist:.3f}",
            os.path.join(OUTPUT_DIR, "test_best_route.png"),
        )

    # ── Worst test route map (for analysis) ─────────────────────────────
    if worst_coords is not None:
        plot_route_map(
            worst_coords.numpy(), worst_actions.tolist(),
            GRAPH_SIZE, worst_dist,
            f"Worst Test Route — CVRP-{GRAPH_SIZE} | Tour: {worst_dist:.3f}",
            os.path.join(OUTPUT_DIR, "test_worst_route.png"),
        )

    print("\n  Done.")


if __name__ == "__main__":
    main()
