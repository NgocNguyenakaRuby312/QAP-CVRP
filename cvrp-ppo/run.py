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

    # Tour length
    T = actions.shape[1]
    idx   = actions.unsqueeze(-1).expand(B, T, 2)
    route = coords.to(device).gather(1, idx)
    depot = coords[:, 0:1, :].to(device)
    full  = torch.cat([depot, route, depot], dim=1)
    dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)

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
        num_samples=min(opts.val_size, 256),
        graph_size=opts.graph_size,
        seed=opts.seed + 1,
        device=str(device),
    )

    # ── Model ────────────────────────────────────────────────────────
    policy = QAPPolicy(
        feature_dim=5, hidden_dim=opts.hidden_dim,
        knn_k=opts.knn_k, lambda_init=opts.lambda_init,
    ).to(device)
    critic = CVRPCritic(feature_dim=5, hidden_dim=opts.hidden_dim).to(device)

    actor_p  = sum(p.numel() for p in policy.parameters())
    critic_p = sum(p.numel() for p in critic.parameters())
    print(f"Model: {actor_p + critic_p} params (actor={actor_p}, critic={critic_p})")

    # ── Generator + Env + Trainer ────────────────────────────────────
    gen = CVRPGenerator(num_loc=opts.graph_size, capacity=opts.capacity)
    env = CVRPEnv(num_loc=opts.graph_size, device=str(device))

    trainer = PPOTrainer(
        policy=policy, critic=critic, env=env, generator=gen,
        clip_epsilon=opts.eps_clip, entropy_coef=opts.c2,
        value_coef=opts.c1, max_grad_norm=opts.max_grad_norm,
        ppo_epochs=opts.K_epochs, gamma=opts.gamma,
        gae_lambda=opts.gae_lambda, lr=opts.lr_model,
        batch_size=opts.batch_size, device=str(device),
        log_dir=opts.output_dir,
    )

    os.makedirs(opts.output_dir, exist_ok=True)
    iters_per_epoch = opts.epoch_size // opts.batch_size

    # ── Header ───────────────────────────────────────────────────────
    print(f"\n{'Epoch':>5} | {'Tour Len':>9} | {'Feas%':>6} | "
          f"{'PPO Loss':>9} | {'Entropy':>8} | {'VRAM MB':>8} | {'Time':>6}")
    print("-" * 72)

    best_val = float("inf")
    for epoch in range(opts.n_epochs):
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

        if val_feas < 0.95:
            print(f"\n*** Feasibility dropped below 95% ({val_feas*100:.1f}%) — stopping ***")
            break

        # ── Checkpoint ───────────────────────────────────────────────
        ckpt_path = os.path.join(opts.output_dir, f"epoch_{epoch+1:03d}.pt")
        torch.save({
            "epoch": epoch + 1,
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
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
