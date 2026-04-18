"""
utils/metrics.py  —  Solution quality metrics
utils/logger.py   —  Training logger: JSON-lines file + optional W&B / TensorBoard
utils/checkpoint.py — Save / load checkpoints
"""

# ═══════════════════════════════════════════════════════════════════════════
# metrics.py
# ═══════════════════════════════════════════════════════════════════════════

import os
import torch


def compute_total_distance(state: dict, actions: torch.Tensor) -> torch.Tensor:
    """
    Total route distance per instance (all edges, depot → … → depot).

    Args:
        state:   env state with key 'coords' [B, N+1, 2]
        actions: LongTensor [B, T]

    Returns:
        dist: FloatTensor [B]
    """
    coords = state["coords"]
    B, T   = actions.shape
    idx    = actions.unsqueeze(-1).expand(B, T, 2)
    route  = coords.gather(1, idx)
    depot  = coords[:, 0:1, :]
    full   = torch.cat([depot, route, depot], dim=1)
    return (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(dim=-1)


def optimality_gap(sol_dist: torch.Tensor, opt_dist: torch.Tensor) -> torch.Tensor:
    """Gap (%) = 100 × (sol − opt) / opt."""
    return 100.0 * (sol_dist - opt_dist) / opt_dist.clamp(min=1e-8)


def feasibility_rate(routes: list, demands: torch.Tensor, capacity: float) -> float:
    """Fraction of routes satisfying Σdᵢ ≤ C (not used in batched training)."""
    ok = sum(1 for r in routes if demands[r].sum() <= capacity)
    return ok / max(len(routes), 1)


def compute_metrics(state: dict, actions: torch.Tensor, opt_dist=None) -> dict:
    """Convenience wrapper returning a dict of scalar metrics."""
    dist = compute_total_distance(state, actions)
    out  = {"mean_dist": dist.mean().item(), "std_dist": dist.std().item()}
    if opt_dist is not None:
        gap = optimality_gap(dist, opt_dist)
        out["mean_gap"] = gap.mean().item()
    return out


# ═══════════════════════════════════════════════════════════════════════════
# logger.py
# ═══════════════════════════════════════════════════════════════════════════

import json
import time


# Mapping: log key → human label for the console summary line
_FIELD_LABELS = {
    "val_tour":     "val_tour",
    "val_gap_pct":  "gap%",
    "train_tour":   "train_tour",
    "greedy_tour":  "greedy",
    "improvement":  "imprv",
    "policy_loss":  "p_loss",
    "value_loss":   "v_loss",
    "entropy":      "entropy",
    "grad_norm":    "grad",
    "clip_fraction":"clip%",
    "ratio_mean":   "ratio",
    "adv_std":      "adv_std",
    "lambda_val":   "λ",
    "lr":           "lr",
    "feasibility":  "feas%",
    "vram_mb":      "vram",
    "epoch_time_s": "t(s)",
}


class Logger:
    """
    Training logger.

    Writes one JSON line per epoch to train_log.jsonl.
    Each line contains ~20 fields covering every failure mode:

        Validation quality:   val_tour, val_gap_pct, best_tour, best_epoch
        Training quality:     train_tour, greedy_tour, improvement
        PPO losses (SPLIT):   policy_loss, value_loss, entropy
        Gradient health:      grad_norm, clip_fraction, ratio_mean
        Learning signal:      adv_mean, adv_std
        Model state:          lambda_val, lr
        System:               feasibility, vram_mb, epoch_time_s

    Why each field matters for diagnosis:
        policy_loss   — should decrease; flat = no gradient signal
        value_loss    — should fall fast then stay near 0; high = critic diverged
        entropy       — should decrease from ~3.5; flat = policy not committing
        grad_norm     — >10 explosion, <1e-4 vanishing
        clip_fraction — >0.3 policy changing too fast (reduce LR)
        ratio_mean    — should stay near 1.0
        adv_std       — near-zero = greedy and sampled indistinguishable (no signal)
        improvement   — greedy_tour − train_tour; near-0 = policy == random
        lambda_val    — should drift from 0.1 as it learns interference weight
        lr            — confirms cosine schedule is moving

    Args:
        log_dir:         directory for train_log.jsonl
        use_wandb:       enable W&B logging
        use_tensorboard: enable TensorBoard SummaryWriter
        project:         W&B project name
        run_name:        W&B / TB run identifier
    """

    def __init__(
        self,
        log_dir:         str  = "outputs",
        use_wandb:       bool = False,
        use_tensorboard: bool = False,
        project:         str  = "cvrp-ppo",
        run_name:        str  = None,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "train_log.jsonl")
        self._file    = open(self.log_path, "a")

        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(project=project, name=run_name)
            except ImportError:
                print("[Logger] wandb not installed; skipping.")

        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("[Logger] TensorBoard not installed; skipping.")

    def log_scalars(self, metrics: dict, step: int):
        """
        Write one epoch record to train_log.jsonl.

        Args:
            metrics: dict of scalar values — should include all fields
                     listed in _FIELD_LABELS for full diagnostic coverage
            step:    epoch number (1-indexed)
        """
        row = {"step": step, "time": time.time(), **metrics}
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

        if self._wandb:
            self._wandb.log(metrics, step=step)

        if self._tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)

    def close(self):
        self._file.close()
        if self._tb:
            self._tb.close()
        if self._wandb:
            self._wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════
# checkpoint.py
# ═══════════════════════════════════════════════════════════════════════════


def save_checkpoint(policy, critic, optimizer, iteration: int, metrics: dict, path: str):
    """Save full training state to disk."""
    torch.save(
        {
            "iteration":       iteration,
            "policy_state":    policy.state_dict(),
            "critic_state":    critic.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics":         metrics,
        },
        path,
    )
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(path: str, policy, critic, optimizer) -> int:
    """Restore training state. Returns the saved iteration number."""
    ckpt = torch.load(path, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state"])
    critic.load_state_dict(ckpt["critic_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[Checkpoint] Loaded ← {path}  (iteration {ckpt['iteration']})")
    return ckpt["iteration"]


def save_best_model(policy, metric_val: float, best_val: float, path: str) -> float:
    """Save policy weights only if metric_val is better (lower) than best_val."""
    if metric_val < best_val:
        torch.save(policy.state_dict(), path)
        print(f"[Best model] Updated ({best_val:.4f} → {metric_val:.4f}) → {path}")
        return metric_val
    return best_val
