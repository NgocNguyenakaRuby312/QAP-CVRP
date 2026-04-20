#!/usr/bin/env python
"""
train_ablation_n20.py — Ablation study: QAP-DRL vs Pure DRL baseline (CVRP-20).

Runs BOTH models back-to-back under identical conditions and saves a
side-by-side comparison chart.  This is the core evidence for/against
the quantum-amplitude contribution.

Ablation variant (b) from thesis §3.4.1:
    "Remove amplitude projection + rotation; use raw 5D features as embedding"
    → encoder_type="baseline" in QAPPolicy

Results table printed at end:
    Model         | Best tour | Gap vs ORT | Gap vs baseline | Params
    QAP-DRL       | X.XXXX   | +XX.X%    | —              | 391
    Pure DRL (b)  | X.XXXX   | +XX.X%    | +/-X.X%        | 278

If QAP-DRL gap < baseline gap  → quantum components contribute positively
If QAP-DRL gap > baseline gap  → quantum components HURT (important finding)
If difference < 0.5%           → no meaningful contribution (null result)

All three outcomes are publishable in a thesis.

Usage:
    cd cvrp-ppo
    python train_ablation_n20.py
"""

import os, sys, time, json, shutil
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

from tqdm import tqdm
from models.qap_policy import QAPPolicy
from environment.cvrp_env import CVRPEnv
from training.ppo_agent import PPOTrainer
from training.evaluate import evaluate_augmented
from utils.seed import set_seed
from utils.data_generator import generate_instances, load_dataset
from utils.ortools_refs import ensure_ortools_ref

# ═══════════════════════════════════════════════════════════════════════════
# Settings — identical for BOTH models
# ═══════════════════════════════════════════════════════════════════════════
GRAPH_SIZE        = 20
CAPACITY          = 30
BATCH_SIZE        = 512
N_EPOCHS          = 200
EPOCH_SIZE        = 128_000
LR                = 1e-4
ENTROPY_COEF      = 0.01
VALUE_COEF        = 0.5
KNN_K             = 10
SEED              = 1234
AUG_SAMPLES       = 8
BATCHES_PER_EPOCH = EPOCH_SIZE // BATCH_SIZE           # 250
TOTAL_OPT_STEPS   = N_EPOCHS * BATCHES_PER_EPOCH * 3 * 8
VAL_EVAL_SIZE     = 500
VAL_PATH          = os.path.join(SCRIPT_DIR, "datasets", "val_n20.pkl")
OUTPUT_DIR        = os.path.join(SCRIPT_DIR, "outputs", "ablation_n20")
ORTOOLS_TIME_LIMIT = 2.0


class CVRPGenerator:
    def __init__(self, graph_size, capacity):
        self.num_loc = graph_size; self.capacity = capacity

    def generate(self, batch_size, device="cpu"):
        coords, demands, cap = generate_instances(batch_size, self.num_loc, self.capacity, device)
        return {"coords": coords, "demands": demands, "capacity": cap}


def _avg(lst, key):
    return sum(l[key] for l in lst) / len(lst)


def run_model(encoder_type: str, device, val_instances, ORTOOLS_REF: float) -> dict:
    """
    Train one model variant for N_EPOCHS and return result dict.

    Args:
        encoder_type: "qap" or "baseline"
        device:       torch device
        val_instances: (coords, demands, cap) on device
        ORTOOLS_REF:  mean OR-Tools tour length

    Returns:
        dict with keys: tour_hist, best_tour, best_epoch, n_params, label
    """
    label = "QAP-DRL (full)" if encoder_type == "qap" else "Pure DRL baseline (no QAP)"
    out_subdir = os.path.join(OUTPUT_DIR, encoder_type)
    os.makedirs(out_subdir, exist_ok=True)

    set_seed(SEED)  # identical seed for both models

    policy    = QAPPolicy(knn_k=KNN_K, encoder_type=encoder_type)
    env       = CVRPEnv(num_loc=GRAPH_SIZE, device=str(device))
    generator = CVRPGenerator(GRAPH_SIZE, CAPACITY)
    n_params  = sum(p.numel() for p in policy.parameters())

    trainer = PPOTrainer(
        policy=policy, env=env, generator=generator,
        lr=LR, entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF,
        batch_size=BATCH_SIZE, total_steps=TOTAL_OPT_STEPS,
        device=str(device), log_dir=out_subdir,
    )

    print(f"\n  {'='*68}")
    print(f"  Training: {label}")
    print(f"  Parameters: {n_params}  |  Encoder: {encoder_type}")
    print(f"  {'='*68}")

    tour_hist  = []
    best_tour  = float("inf")
    best_epoch = 0

    epoch_bar = tqdm(range(1, N_EPOCHS + 1), desc=f"{encoder_type:8s}", unit="ep")
    for epoch in epoch_bar:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        losses_list = []
        for _ in range(BATCHES_PER_EPOCH):
            trainer.collect_rollout()
            losses_list.append(trainer.update())

        val_tour    = evaluate_augmented(trainer.policy, val_instances, device, n_samples=AUG_SAMPLES)
        val_gap_ort = 100.0 * (val_tour - ORTOOLS_REF) / ORTOOLS_REF

        is_best = val_tour < best_tour
        if is_best:
            best_tour = val_tour; best_epoch = epoch
            torch.save(trainer.policy.state_dict(), os.path.join(out_subdir, "best_model.pt"))

        tour_hist.append(val_tour)
        epoch_bar.set_postfix(best=f"{best_tour:.4f}", gap=f"{val_gap_ort:+.1f}%")

        # Save per-epoch log line
        with open(os.path.join(out_subdir, "log.jsonl"), "a") as f:
            json.dump({"epoch": epoch, "val_tour": val_tour, "gap": val_gap_ort,
                       "best_tour": best_tour, "entropy": _avg(losses_list, "entropy"),
                       "adv_std": _avg(losses_list, "adv_std"),
                       "clip_fraction": _avg(losses_list, "clip_fraction")}, f)
            f.write("\n")

    trainer.logger.close()
    print(f"\n  {label}: best tour = {best_tour:.4f}  "
          f"(gap = {100*(best_tour-ORTOOLS_REF)/ORTOOLS_REF:+.2f}%)  at epoch {best_epoch}")

    return {
        "tour_hist":  tour_hist,
        "best_tour":  best_tour,
        "best_epoch": best_epoch,
        "n_params":   n_params,
        "label":      label,
        "encoder_type": encoder_type,
    }


def plot_comparison(results: list, ORTOOLS_REF: float, save_path: str):
    """Side-by-side tour length curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Ablation Study — CVRP-{GRAPH_SIZE}: QAP-DRL vs Pure DRL Baseline", fontsize=13)

    colors = {"qap": "#378ADD", "baseline": "#D85A30"}
    epochs = list(range(1, N_EPOCHS + 1))

    # Left: both curves on same axis
    ax = axes[0]
    for r in results:
        c = colors[r["encoder_type"]]
        ax.plot(epochs, r["tour_hist"], color=c, lw=1.5,
                label=f"{r['label']} (best={r['best_tour']:.4f}, {r['n_params']}p)")
    ax.axhline(y=ORTOOLS_REF, color="darkorange", ls="--", lw=1.2,
               label=f"OR-Tools ({ORTOOLS_REF:.4f})")
    ax.set_title("Tour length comparison"); ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg distance (aug×8)"); ax.legend(fontsize=8)

    # Right: gap relative to baseline at each epoch
    ax2 = axes[1]
    if len(results) == 2:
        qap_hist = results[0]["tour_hist"] if results[0]["encoder_type"] == "qap" else results[1]["tour_hist"]
        bl_hist  = results[1]["tour_hist"] if results[1]["encoder_type"] == "baseline" else results[0]["tour_hist"]
        gap_diff = [100 * (q - b) / b for q, b in zip(qap_hist, bl_hist)]
        ax2.plot(epochs, gap_diff, color="#534AB7", lw=1.5)
        ax2.axhline(y=0, color="black", ls="--", lw=1.0, alpha=0.5)
        ax2.fill_between(epochs, gap_diff, 0,
                         where=[g < 0 for g in gap_diff], color="#1D9E75", alpha=0.2,
                         label="QAP-DRL better")
        ax2.fill_between(epochs, gap_diff, 0,
                         where=[g > 0 for g in gap_diff], color="#D85A30", alpha=0.2,
                         label="Baseline better")
        ax2.set_title("QAP-DRL gap vs baseline (%)\n(negative = QAP-DRL is better)")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("(QAP-DRL − baseline) / baseline × 100%")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Comparison chart saved → {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    val_coords, val_demands, val_cap = load_dataset(VAL_PATH, device="cpu")
    ORTOOLS_REF, ORTOOLS_SOURCE = ensure_ortools_ref(
        n=GRAPH_SIZE, val_path=VAL_PATH, n_instances=VAL_EVAL_SIZE,
        coords_t=val_coords, demands_t=val_demands, capacity=val_cap,
        time_limit=ORTOOLS_TIME_LIMIT, output_dir=OUTPUT_DIR,
    )

    val_instances = (
        val_coords[:VAL_EVAL_SIZE].to(device),
        val_demands[:VAL_EVAL_SIZE].to(device),
        val_cap,
    )

    print(f"\n  QAP-DRL Ablation Study — CVRP-{GRAPH_SIZE}")
    print(f"  Both models use identical hyperparams, seed={SEED}")
    print(f"  OR-Tools reference: {ORTOOLS_REF:.4f}")
    print(f"  Output dir: {OUTPUT_DIR}\n")

    results = []
    for enc in ["qap", "baseline"]:
        results.append(run_model(enc, device, val_instances, ORTOOLS_REF))

    # ── Results table ────────────────────────────────────────────────────
    print(f"\n  {'='*72}")
    print(f"  ABLATION RESULTS — CVRP-{GRAPH_SIZE}")
    print(f"  {'='*72}")
    print(f"  {'Model':<30} | {'Best tour':>9} | {'Gap vs ORT':>10} | {'Params':>7} | {'Best ep':>7}")
    print(f"  {'-'*70}")

    baseline_best = None
    for r in results:
        gap = 100 * (r["best_tour"] - ORTOOLS_REF) / ORTOOLS_REF
        if r["encoder_type"] == "baseline":
            baseline_best = r["best_tour"]
        vs_bl = ""
        if r["encoder_type"] == "qap" and baseline_best is not None:
            diff = 100 * (r["best_tour"] - baseline_best) / baseline_best
            vs_bl = f"  ({diff:+.2f}% vs baseline)"
        print(f"  {r['label']:<30} | {r['best_tour']:>9.4f} | {gap:>+9.2f}% | "
              f"{r['n_params']:>7} | {r['best_epoch']:>7}{vs_bl}")

    print(f"  {'-'*70}")
    if len(results) == 2:
        q = next(r for r in results if r["encoder_type"] == "qap")
        b = next(r for r in results if r["encoder_type"] == "baseline")
        diff = 100 * (q["best_tour"] - b["best_tour"]) / b["best_tour"]
        if diff < -0.5:
            verdict = f"QAP-DRL is BETTER by {abs(diff):.2f}% — quantum components contribute positively"
        elif diff > 0.5:
            verdict = f"Baseline is BETTER by {abs(diff):.2f}% — quantum encoding HURTS (important finding)"
        else:
            verdict = f"No meaningful difference ({diff:+.2f}%) — quantum components are neutral"
        print(f"  Verdict: {verdict}")
    print(f"  {'='*72}\n")

    plot_comparison(results, ORTOOLS_REF,
                    os.path.join(OUTPUT_DIR, "ablation_comparison.png"))

    # Save results JSON
    summary = {
        "graph_size": GRAPH_SIZE, "ortools_ref": ORTOOLS_REF,
        "models": [{k: v for k, v in r.items() if k != "tour_hist"} for r in results],
        "verdict": verdict if len(results) == 2 else "incomplete",
    }
    with open(os.path.join(OUTPUT_DIR, "ablation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {os.path.join(OUTPUT_DIR, 'ablation_results.json')}")


if __name__ == "__main__":
    main()
