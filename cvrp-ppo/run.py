"""
run.py — Main training entry point for QAP-DRL on CVRP.

Usage:
    python run.py
    python run.py --graph_size 50 --n_epochs 200
    python run.py --no_cuda
"""

import os
import sys
import time
import math
import torch

sys.path.insert(0, os.path.dirname(__file__))

from options import get_options
from utils.seed import set_seed
from utils.data_generator import generate_instances, generate_validation_set
from environment.cvrp_env import CVRPEnv
from models.qap_policy import QAPPolicy, CVRPCritic
from training.ppo_agent import PPOTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Spec-compliant data generator for PPOTrainer interface
# ─────────────────────────────────────────────────────────────────────────────

class CVRPGenerator:
    """Generates CVRP instances per Kool et al. 2019 protocol (§7)."""

    def __init__(self, num_loc: int, capacity: int):
        self.num_loc  = num_loc
        self.capacity = capacity

    def generate(self, batch_size: int, device: str = "cpu") -> dict:
        B, N, C = batch_size, self.num_loc, self.capacity
        coords  = torch.FloatTensor(B, N + 1, 2).uniform_(0, 1).to(device)  # [B, N+1, 2]
        demands = torch.zeros(B, N + 1, device=device)                       # [B, N+1]
        demands[:, 1:] = torch.randint(1, 10, (B, N), device=device).float() # {1..9}
        capacity = torch.full((B,), float(C), device=device)                  # [B]
        return {"coords": coords, "demands": demands, "capacity": capacity}


# ─────────────────────────────────────────────────────────────────────────────
# Feasibility checker
# ─────────────────────────────────────────────────────────────────────────────

def check_feasibility(actions, demands, capacity):
    """Return fraction of batch elements with feasible tours."""
    B = actions.shape[0]
    feasible = 0
    for b in range(B):
        cap = float(capacity)
        ok = True
        for a in actions[b]:
            n = a.item()
            if n == 0:
                cap = float(capacity)
            else:
                cap -= demands[b, n].item()
                if cap < -1e-6:
                    ok = False
                    break
        if ok:
            feasible += 1
    return feasible / B


# ─────────────────────────────────────────────────────────────────────────────
# Greedy evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_eval(policy, env, coords, demands, capacity, device):
    """Run greedy decoding on validation set, return (mean_tour_length, feasibility_rate)."""
    policy.eval()
    B = coords.size(0)
    state = env.reset({
        "coords":   coords.to(device),
        "demands":  demands.to(device),
        "capacity": torch.full((B,), float(capacity), device=device),
    })
    actions, _, _ = policy(state, env, deterministic=True)

    dists = -env.get_reward(state, actions)                            # get_reward returns −distance
    feas = check_feasibility(actions, demands.to(device), capacity)
    return dists.mean().item(), feas


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    opts = get_options()
    set_seed(opts.seed)

    # ── Device (detect HERE and only here) ───────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not opts.no_cuda else "cpu"
    )
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"Problem: N={opts.graph_size}, C={opts.capacity}, seed={opts.seed}")
    print(f"Training: batch={opts.batch_size}, epochs={opts.n_epochs}, "
          f"epoch_size={opts.epoch_size}, lr={opts.lr_model}")

    # ── Validation data (fixed) ──────────────────────────────────────
    val_coords, val_demands, val_cap = generate_validation_set(
        num_samples=min(opts.val_size, opts.eval_batch_size),
        graph_size=opts.graph_size,
        seed=opts.seed + 1,
        device=str(device),
    )

    # ── Model ────────────────────────────────────────────────────────
    # FIXED: QAPPolicy already embeds critic_head (§5: ~391 total params).
    # Previously a separate CVRPCritic was also instantiated, adding 257 duplicate
    # params (total ~648). Now policy alone is used — matches the spec budget.
    policy = QAPPolicy(
        feature_dim=5, hidden_dim=opts.hidden_dim,
        knn_k=opts.knn_k, lambda_init=opts.lambda_init,
    ).to(device)

    total_p = sum(p.numel() for p in policy.parameters())
    print(f"Model: {total_p} params (spec ~391)")

    # ── Generator + Env + Trainer ────────────────────────────────────
    gen = CVRPGenerator(num_loc=opts.graph_size, capacity=opts.capacity)
    env = CVRPEnv(num_loc=opts.graph_size, device=str(device))

    # FIXED: no longer passing critic= (PPOTrainer now uses policy.get_value())
    trainer = PPOTrainer(
        policy=policy, env=env, generator=gen,
        clip_epsilon=opts.eps_clip, entropy_coef=opts.c2,
        value_coef=opts.c1, max_grad_norm=opts.max_grad_norm,
        ppo_epochs=opts.K_epochs, gamma=opts.gamma,
        gae_lambda=opts.gae_lambda, lr=opts.lr_model,
        batch_size=opts.batch_size, device=str(device),
        log_dir=opts.output_dir, n_minibatches=opts.n_minibatches,
    )

    os.makedirs(opts.output_dir, exist_ok=True)

    # ── FIX 5: checkpoint resume ─────────────────────────────────────
    start_epoch = 0
    if opts.load_path and os.path.isfile(opts.load_path):
        ckpt = torch.load(opts.load_path, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        # FIXED: no separate critic key — critic_head is part of policy state_dict
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        print(f"[Resume] Loaded from {opts.load_path} (epoch {start_epoch})")

    # ── FIX 5: eval-only exit ────────────────────────────────────────
    if opts.eval_only:
        val_dist, val_feas = greedy_eval(
            policy, env, val_coords, val_demands, val_cap, device
        )
        print(f"Eval: tour_len={val_dist:.4f}  feasibility={val_feas*100:.1f}%")
        return

    iters_per_epoch = opts.epoch_size // opts.batch_size

    # ── Header ───────────────────────────────────────────────────────
    print(f"\n{'Epoch':>5} | {'Tour Len':>9} | {'Feas%':>6} | "
          f"{'PPO Loss':>9} | {'Entropy':>8} | {'VRAM MB':>8} | {'Time':>6}")
    print("-" * 72)

    best_val = float("inf")
    for epoch in range(start_epoch, opts.n_epochs):
        t0 = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Collect + Update ─────────────────────────────────────────
        epoch_reward = 0.0
        losses = {}
        for it in range(iters_per_epoch):
            reward = trainer.collect_rollout()
            losses = trainer.update()
            epoch_reward += reward

            # NaN check
            if math.isnan(losses.get("policy_loss", 0)):
                print(f"\n*** NaN detected in PPO loss at epoch {epoch+1}, iter {it} ***")
                return

        epoch_reward /= iters_per_epoch
        elapsed = time.time() - t0

        # ── Greedy validation ────────────────────────────────────────
        val_dist, val_feas = greedy_eval(
            policy, env, val_coords, val_demands, val_cap, device
        )

        # ── VRAM ─────────────────────────────────────────────────────
        vram_mb = 0.0
        if device.type == "cuda":
            vram_mb = torch.cuda.memory_allocated() / 1024**2

        # ── Print ────────────────────────────────────────────────────
        print(f"{epoch+1:5d} | {val_dist:9.4f} | {val_feas*100:5.1f}% | "
              f"{losses.get('policy_loss',0):9.4f} | "
              f"{losses.get('entropy',0):8.4f} | "
              f"{vram_mb:7.1f} | {elapsed:5.1f}s")

        # ── Safety checks ────────────────────────────────────────────
        if vram_mb > 3500:
            print(f"\n*** VRAM exceeded 3.5 GB ({vram_mb:.0f} MB) — stopping ***")
            break

        if epoch >= 10 and val_feas < 0.95:
            print(f"\n*** Feasibility dropped below 95% ({val_feas*100:.1f}%) — stopping ***")
            break

        # ── Checkpoint ───────────────────────────────────────────────
        ckpt_path = os.path.join(opts.output_dir, f"epoch_{epoch+1:03d}.pt")
        # FIXED: no separate critic state — critic_head is inside policy
        torch.save({
            "epoch": epoch + 1,
            "policy": policy.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "val_dist": val_dist,
            "val_feas": val_feas,
        }, ckpt_path)

        if val_dist < best_val:
            best_val = val_dist
            torch.save(policy.state_dict(), os.path.join(opts.output_dir, "best.pt"))

    print(f"\nBest validation tour length: {best_val:.4f}")


if __name__ == "__main__":
    main()
