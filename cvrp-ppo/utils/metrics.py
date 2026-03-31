"""
utils/metrics.py  —  Solution quality metrics
utils/logger.py   —  Training logger (console + optional W&B / TensorBoard)
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


class Logger:
    """
    Lightweight logger.  Writes JSON lines to a file and optionally
    forwards to Weights & Biases or TensorBoard.

    Args:
        log_dir:         directory for log files
        use_wandb:       enable W&B logging
        use_tensorboard: enable TensorBoard logging
        project:         W&B project name
        run_name:        W&B / TB run name
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
        """Write metrics to file, W&B, and TensorBoard."""
        row = {"step": step, "time": time.time(), **metrics}
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

        if self._wandb:
            self._wandb.log(metrics, step=step)

        if self._tb:
            for k, v in metrics.items():
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
    """Save policy weights only if metric_val is better than best_val."""
    if metric_val > best_val:
        torch.save(policy.state_dict(), path)
        print(f"[Best model] Updated ({best_val:.4f} → {metric_val:.4f}) → {path}")
        return metric_val
    return best_val
