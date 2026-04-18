#!/usr/bin/env python
"""
generate_thesis_results.py
===========================
Generates thesis-ready outputs after training completes:

  1. Performance comparison TABLE (LaTeX) — like Q-GAT Table 2
  2. Training convergence CHART           — like Q-GAT Fig. 3
  3. Parameter count TABLE (LaTeX)        — like Q-GAT Table 1

Usage:
    python generate_thesis_results.py                    # all sizes
    python generate_thesis_results.py --sizes 20 50      # specific sizes
    python generate_thesis_results.py --skip-baselines   # QAP-DRL only
"""

import os, sys, json, argparse, math, time
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

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

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
CONFIGS = {
    20:  {"capacity": 30, "lkh3": 6.10,  "val": "datasets/val_n20.pkl",
          "model": "outputs/n20/best_model.pt",  "log": "outputs/n20/train_log.jsonl"},
    50:  {"capacity": 40, "lkh3": 10.38, "val": "datasets/val_n50.pkl",
          "model": "outputs/n50/best_model.pt",  "log": "outputs/n50/train_log.jsonl"},
    100: {"capacity": 50, "lkh3": 15.65, "val": "datasets/val_n100.pkl",
          "model": "outputs/n100/best_model.pt", "log": "outputs/n100/train_log.jsonl"},
}

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "thesis_results")

# Number of sampled solutions per instance for sampling decode
N_SAMPLES = 8


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_model(policy, coords, demands, capacity, n, device,
                   greedy=True, n_samples=1):
    """
    Evaluate QAP-DRL on instances.

    Args:
        greedy:    True = argmax (deterministic), False = sampling
        n_samples: if sampling, run n_samples times and take best per instance

    Returns:
        avg_length: float
        ms_per_inst: float
    """
    policy.eval()
    B = coords.size(0)
    BATCH = 256

    all_dists = []
    t0 = time.time()

    for i in range(0, B, BATCH):
        j = min(i + BATCH, B)
        bc = coords[i:j].to(device)
        bd = demands[i:j].to(device)
        bsize = bc.size(0)

        env = CVRPEnv(num_loc=n, device=str(device))

        if greedy or n_samples <= 1:
            state = env.reset({
                "coords": bc, "demands": bd,
                "capacity": torch.full((bsize,), float(capacity), device=device),
            })
            actions, _, _ = policy(state, env, deterministic=greedy)
            T = actions.shape[1]
            idx = actions.unsqueeze(-1).expand(bsize, T, 2)
            route = bc.gather(1, idx)
            depot = bc[:, 0:1, :]
            full = torch.cat([depot, route, depot], dim=1)
            dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)
            all_dists.append(dists)
        else:
            # Sampling: run n_samples times, keep best per instance
            best_dists = torch.full((bsize,), float("inf"), device=device)
            for _ in range(n_samples):
                state = env.reset({
                    "coords": bc, "demands": bd,
                    "capacity": torch.full((bsize,), float(capacity), device=device),
                })
                actions, _, _ = policy(state, env, deterministic=False)
                T = actions.shape[1]
                idx = actions.unsqueeze(-1).expand(bsize, T, 2)
                route = bc.gather(1, idx)
                depot = bc[:, 0:1, :]
                full = torch.cat([depot, route, depot], dim=1)
                dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)
                best_dists = torch.min(best_dists, dists)
            all_dists.append(best_dists)

    total_time = time.time() - t0
    all_dists = torch.cat(all_dists)
    return all_dists.mean().item(), (total_time / B) * 1000


def evaluate_nn_baseline(coords, demands, capacity, n):
    """Nearest-neighbour heuristic baseline."""
    B = coords.size(0)
    all_dists = []

    for b in range(B):
        c = coords[b].numpy()              # [N+1, 2]
        d = demands[b].numpy()             # [N+1]
        visited = [False] * (n + 1)
        visited[0] = True                  # depot is "home"
        cur = 0
        used = 0.0
        total_dist = 0.0

        for _ in range(n):
            best_node, best_d = -1, float("inf")
            for j in range(1, n + 1):
                if visited[j]:
                    continue
                if used + d[j] > capacity:
                    continue
                dd = math.sqrt((c[cur, 0] - c[j, 0])**2 + (c[cur, 1] - c[j, 1])**2)
                if dd < best_d:
                    best_d = dd
                    best_node = j

            if best_node == -1:
                # Return to depot and reset
                total_dist += math.sqrt((c[cur, 0] - c[0, 0])**2 + (c[cur, 1] - c[0, 1])**2)
                cur = 0
                used = 0.0
                # Find nearest feasible
                for j in range(1, n + 1):
                    if visited[j]:
                        continue
                    dd = math.sqrt((c[0, 0] - c[j, 0])**2 + (c[0, 1] - c[j, 1])**2)
                    if dd < best_d:
                        best_d = dd
                        best_node = j

            total_dist += best_d
            visited[best_node] = True
            used += d[best_node]
            cur = best_node

        # Return to depot
        total_dist += math.sqrt((c[cur, 0] - c[0, 0])**2 + (c[cur, 1] - c[0, 1])**2)
        all_dists.append(total_dist)

    return np.mean(all_dists)


# ═══════════════════════════════════════════════════════════════════════════
# Table generation
# ═══════════════════════════════════════════════════════════════════════════
def generate_comparison_table(sizes, n_instances, skip_baselines, device):
    """
    Generate performance comparison table like Q-GAT Table 2.

    Methods:
        - LKH3 (literature reference)
        - Nearest Neighbour (heuristic)
        - OR-Tools (GLS, if installed)
        - QAP-DRL Greedy
        - QAP-DRL Sampling (×8)
    """
    results = {}

    for n in sizes:
        cfg = CONFIGS[n]
        print(f"\n{'='*50}")
        print(f"  Evaluating CVRP-{n}  ({n_instances} instances)")
        print(f"{'='*50}")

        val_path = os.path.join(SCRIPT_DIR, cfg["val"])
        model_path = os.path.join(SCRIPT_DIR, cfg["model"])

        if not os.path.exists(val_path):
            print(f"  SKIP: val dataset not found ({val_path})")
            continue

        coords, demands, cap = load_dataset(val_path, device="cpu")
        coords = coords[:n_instances]
        demands = demands[:n_instances]

        col = {"lkh3": cfg["lkh3"]}

        # Nearest neighbour
        if not skip_baselines:
            print("  Running Nearest Neighbour ...", end=" ", flush=True)
            nn_len = evaluate_nn_baseline(coords, demands, cfg["capacity"], n)
            col["nn"] = nn_len
            print(f"{nn_len:.2f}")

            # OR-Tools
            try:
                sys.path.insert(0, os.path.join(SCRIPT_DIR, "validation_methods"))
                from ortools_baseline import evaluate_ortools, ORTOOLS_AVAILABLE
                if ORTOOLS_AVAILABLE:
                    print("  Running OR-Tools (GLS, 2s/inst) ...", flush=True)
                    ot_len = evaluate_ortools(n, n_instances, time_limit=2.0)
                    col["ortools"] = ot_len
                    print(f"  OR-Tools: {ot_len:.2f}")
            except Exception as e:
                print(f"  OR-Tools: skipped ({e})")

        # QAP-DRL
        if not os.path.exists(model_path):
            print(f"  SKIP: model not found ({model_path})")
            results[n] = col
            continue

        policy = QAPPolicy()
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.to(device).eval()

        # Greedy
        print("  Running QAP-DRL (Greedy) ...", end=" ", flush=True)
        g_len, g_ms = evaluate_model(
            policy, coords, demands, cfg["capacity"], n, device,
            greedy=True
        )
        col["qap_greedy"] = g_len
        col["qap_greedy_ms"] = g_ms
        print(f"{g_len:.2f}  ({g_ms:.1f} ms/inst)")

        # Sampling ×N_SAMPLES
        print(f"  Running QAP-DRL (Sampling ×{N_SAMPLES}) ...", end=" ", flush=True)
        s_len, s_ms = evaluate_model(
            policy, coords, demands, cfg["capacity"], n, device,
            greedy=False, n_samples=N_SAMPLES
        )
        col["qap_sample"] = s_len
        col["qap_sample_ms"] = s_ms
        print(f"{s_len:.2f}  ({s_ms:.1f} ms/inst)")

        results[n] = col

    return results


def print_table(results, sizes):
    """Print comparison table in console + LaTeX format."""
    lkh3 = {n: CONFIGS[n]["lkh3"] for n in sizes}

    # ── Console table ────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  PERFORMANCE COMPARISON ON CVRP INSTANCES")
    print(f"{'='*78}")

    header = f"  {'Method':<25} {'Type':<8}"
    for n in sizes:
        header += f" {'VRP'+str(n):>12}"
    print(header)
    print(f"  {'':<25} {'':<8}", end="")
    for n in sizes:
        print(f" {'Len':>6} {'Gap':>5}", end="")
    print()
    print(f"  {'-'*74}")

    def print_row(name, typ, values):
        row = f"  {name:<25} {typ:<8}"
        for n in sizes:
            v = values.get(n, None)
            if v is None:
                row += f" {'--':>6} {'--':>5}"
            else:
                gap = 100.0 * (v - lkh3[n]) / lkh3[n]
                row += f" {v:>6.2f} {gap:>4.1f}%"
        print(row)

    # LKH3
    print_row("LKH3", "Solver", {n: lkh3[n] for n in sizes})

    # NN
    nn_vals = {n: results[n].get("nn") for n in sizes if n in results and "nn" in results[n]}
    if nn_vals:
        print_row("Nearest Neighbour", "H, G", nn_vals)

    # OR-Tools
    ot_vals = {n: results[n].get("ortools") for n in sizes if n in results and "ortools" in results[n]}
    if ot_vals:
        print_row("OR-Tools (GLS)", "H, S", ot_vals)

    # QAP-DRL Greedy
    g_vals = {n: results[n].get("qap_greedy") for n in sizes if n in results and "qap_greedy" in results[n]}
    if g_vals:
        print_row("QAP-DRL (Greedy)", "RL, G", g_vals)

    # QAP-DRL Sampling
    s_vals = {n: results[n].get("qap_sample") for n in sizes if n in results and "qap_sample" in results[n]}
    if s_vals:
        print_row(f"QAP-DRL (Sample ×{N_SAMPLES})", "RL, S", s_vals)

    print(f"{'='*78}")

    # ── LaTeX table ──────────────────────────────────────────────────
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Performance comparison on CVRP instances.}")
    latex.append("\\label{tab:results}")

    cols = "ll" + "cc" * len(sizes)
    latex.append(f"\\begin{{tabular}}{{{cols}}}")
    latex.append("\\toprule")

    # Header row 1
    h1 = "Method & Type"
    for n in sizes:
        h1 += f" & \\multicolumn{{2}}{{c}}{{VRP{n}}}"
    h1 += " \\\\"
    latex.append(h1)

    # Header row 2: cmidrules
    for i, n in enumerate(sizes):
        col_start = 3 + i * 2
        latex.append(f"\\cmidrule(lr){{{col_start}-{col_start+1}}}")
    h2 = " & "
    for n in sizes:
        h2 += " & Length & Gap"
    h2 += " \\\\"
    latex.append(h2)
    latex.append("\\midrule")

    def latex_row(name, typ, values, bold=False):
        if bold:
            row = f"\\textbf{{{name}}} & \\textbf{{{typ}}}"
        else:
            row = f"{name} & {typ}"
        for n in sizes:
            v = values.get(n, None)
            if v is None:
                row += " & -- & --"
            else:
                gap = 100.0 * (v - lkh3[n]) / lkh3[n]
                vstr = f"{v:.2f}"
                gstr = f"{gap:.2f}\\%"
                if bold:
                    row += f" & \\textbf{{{vstr}}} & \\textbf{{{gstr}}}"
                else:
                    row += f" & {vstr} & {gstr}"
        row += " \\\\"
        return row

    latex.append(latex_row("LKH3", "Solver", {n: lkh3[n] for n in sizes}))
    latex.append("\\midrule")
    if nn_vals:
        latex.append(latex_row("Nearest Neighbour", "H, G", nn_vals))
    if ot_vals:
        latex.append(latex_row("OR-Tools (GLS)", "H, S", ot_vals))
    if g_vals:
        latex.append(latex_row("QAP-DRL (Greedy)", "RL, G", g_vals, bold=True))
    if s_vals:
        latex.append(latex_row(f"QAP-DRL (Sample $\\times${N_SAMPLES})", "RL, S", s_vals, bold=True))

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)
    print("\n── LaTeX Table ──────────────────────────────────────")
    print(latex_str)

    return latex_str


# ═══════════════════════════════════════════════════════════════════════════
# Training convergence chart (like Q-GAT Fig. 3)
# ═══════════════════════════════════════════════════════════════════════════
def plot_convergence(sizes):
    """
    Plot training and validation tour length over epochs.
    Reads train_log.jsonl files from each size's output directory.
    """
    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 5))
    if len(sizes) == 1:
        axes = [axes]

    colors = {"train": "#1f77b4", "val": "#ff7f0e"}

    for ax, n in zip(axes, sizes):
        cfg = CONFIGS[n]
        log_path = os.path.join(SCRIPT_DIR, cfg["log"])

        if not os.path.exists(log_path):
            ax.set_title(f"CVRP-{n}\n(no training log)")
            continue

        epochs, val_tours, ppo_losses = [], [], []
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                epochs.append(entry["step"])
                val_tours.append(entry["val_tour"])
                ppo_losses.append(entry.get("ppo_loss", 0))

        # Plot validation tour length (primary metric)
        ax.plot(epochs, val_tours, color=colors["val"], linewidth=1.8,
                label="QAP-DRL (val)")

        # LKH3 reference
        ax.axhline(y=cfg["lkh3"], color="black", linestyle="--",
                    linewidth=1.2, label=f"LKH3 ({cfg['lkh3']})")

        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Tour Length", fontsize=12)
        ax.set_title(f"CVRP-{n}", fontsize=14)
        ax.legend(fontsize=10, loc="upper right")

        # Set y-axis range based on data
        ymin = min(min(val_tours), cfg["lkh3"]) * 0.95
        ymax = max(val_tours[:10]) * 1.05 if len(val_tours) > 10 else max(val_tours) * 1.1
        ax.set_ylim(ymin, ymax)

    fig.suptitle("Training Convergence — QAP-DRL", fontsize=15, y=1.02)
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "convergence_chart.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Convergence chart saved: {save_path}")

    return save_path


# ═══════════════════════════════════════════════════════════════════════════
# Parameter count table (like Q-GAT Table 1)
# ═══════════════════════════════════════════════════════════════════════════
def print_param_table():
    """Print parameter count breakdown."""
    policy = QAPPolicy()

    # Count by component
    encoder_params = sum(p.numel() for p in policy.encoder.qap_encoder.parameters())
    decoder_params = sum(p.numel() for p in policy.decoder.parameters())
    critic_params  = sum(p.numel() for p in policy.critic_head.parameters())
    total          = sum(p.numel() for p in policy.parameters())

    # Detailed breakdown
    amp_proj = sum(p.numel() for p in policy.encoder.qap_encoder.amplitude_proj.parameters())
    rot_mlp  = sum(p.numel() for p in policy.encoder.qap_encoder.rotation_mlp.parameters())
    ctx_q    = sum(p.numel() for p in policy.decoder.context_query.parameters())
    hybrid   = sum(p.numel() for p in policy.decoder.hybrid.parameters())

    print(f"\n{'='*55}")
    print("  MODEL COMPLEXITY — QAP-DRL")
    print(f"{'='*55}")
    print(f"  {'Component':<30} {'Parameters':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Amplitude Projection (W,b)':<30} {amp_proj:>10}")
    print(f"  {'Rotation MLP (5→16→1)':<30} {rot_mlp:>10}")
    print(f"  {'Context Query (Wq)':<30} {ctx_q:>10}")
    print(f"  {'Hybrid Scoring (λ)':<30} {hybrid:>10}")
    print(f"  {'Critic Head (2→64→1)':<30} {critic_params:>10}")
    print(f"  {'-'*50}")
    print(f"  {'Actor total':<30} {encoder_params + decoder_params:>10}")
    print(f"  {'Critic total':<30} {critic_params:>10}")
    print(f"  {'TOTAL':<30} {total:>10}")
    print(f"{'='*55}")

    # LaTeX version
    latex = f"""
\\begin{{table}}[t]
\\centering
\\caption{{Model complexity of QAP-DRL.}}
\\label{{tab:params}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Model & Actor params & Critic params & Total \\\\
\\midrule
QAP-DRL (ours) & {encoder_params + decoder_params} & {critic_params} & {total} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    print("\n── LaTeX Table ──")
    print(latex)

    return total


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate thesis results")
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 100],
                        choices=[20, 50, 100])
    parser.add_argument("--instances", type=int, default=1000,
                        help="Number of val instances to evaluate")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip NN and OR-Tools baselines")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  QAP-DRL THESIS RESULTS GENERATOR")
    print("=" * 60)
    print(f"  Sizes      : {args.sizes}")
    print(f"  Instances  : {args.instances}")
    print(f"  Baselines  : {'skip' if args.skip_baselines else 'NN + OR-Tools'}")
    print(f"  Device     : {device}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print("=" * 60)

    # 1. Parameter table
    print_param_table()

    # 2. Performance comparison
    results = generate_comparison_table(
        args.sizes, args.instances, args.skip_baselines, device
    )
    latex_table = print_table(results, args.sizes)

    # Save LaTeX
    with open(os.path.join(OUTPUT_DIR, "table_comparison.tex"), "w") as f:
        f.write(latex_table)
    print(f"\n  LaTeX table saved: {os.path.join(OUTPUT_DIR, 'table_comparison.tex')}")

    # Save JSON
    json_results = {}
    for n, col in results.items():
        json_results[str(n)] = {k: v for k, v in col.items() if v is not None}
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    # 3. Convergence chart
    plot_convergence(args.sizes)

    print(f"\n{'='*60}")
    print("  ALL RESULTS GENERATED")
    print(f"  Check: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
