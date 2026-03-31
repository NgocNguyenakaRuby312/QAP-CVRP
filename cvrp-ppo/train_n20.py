#!/usr/bin/env python
"""train_n20.py — Train QAP-DRL on CVRP-20."""

import os, sys, time
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

from models.qap_policy import QAPPolicy, CVRPCritic
from environment.cvrp_env import CVRPEnv
from training.ppo_agent import PPOTrainer
from training.evaluate import evaluate
from utils.seed import set_seed
from utils.data_generator import generate_instances, load_dataset

# ═══════════════════════════════════════════════════════════════════════════
# Settings (hardcoded — no CLI args needed)
# ═══════════════════════════════════════════════════════════════════════════
GRAPH_SIZE        = 20
CAPACITY          = 30
BATCH_SIZE        = 256
N_EPOCHS          = 100
EPOCH_SIZE        = 128_000
LR                = 1e-4
SEED              = 1234
LKH3_REF          = 6.10
BATCHES_PER_EPOCH = EPOCH_SIZE // BATCH_SIZE   # 500
OUTPUT_DIR        = os.path.join(SCRIPT_DIR, "outputs", "n20")
VAL_PATH          = os.path.join(SCRIPT_DIR, "datasets", "val_n20.pkl")
VAL_EVAL_SIZE     = 1000


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
class CVRPGenerator:
    """Minimal wrapper so PPOTrainer.collect_rollout() can call .generate()."""
    def __init__(self, graph_size, capacity):
        self.num_loc = graph_size
        self.capacity = capacity

    def generate(self, batch_size, device="cpu"):
        coords, demands, cap = generate_instances(
            batch_size, self.num_loc, self.capacity, device
        )
        return {"coords": coords, "demands": demands, "capacity": cap}


def check_feasibility(demands, capacity, actions):
    """Vectorised feasibility check. Returns fraction of feasible routes."""
    B, T = actions.shape
    used = torch.zeros(B, device=actions.device)
    violated = torch.zeros(B, dtype=torch.bool, device=actions.device)
    for t in range(T):
        at_depot = (actions[:, t] == 0)
        used = torch.where(at_depot, torch.zeros_like(used), used)
        d = demands.gather(1, actions[:, t:t + 1]).squeeze(1)
        used = used + d
        violated = violated | (used > capacity + 1e-6)
    return (~violated).float().mean().item()


def plot_route_map(coords_np, actions_list, graph_size, tour_length,
                   title, save_path):
    """Plot a single CVRP solution with coloured vehicle routes and arrows."""
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Split action sequence into vehicle sub-routes
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

    # Depot
    ax.plot(coords_np[0, 0], coords_np[0, 1], "r*",
            markersize=20, label="Depot", zorder=5)

    # Customers
    for i in range(1, graph_size + 1):
        ax.plot(coords_np[i, 0], coords_np[i, 1], "o",
                color="#4a90d9", markersize=6, zorder=4)
        ax.annotate(str(i), (coords_np[i, 0], coords_np[i, 1]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)

    # Vehicle routes with direction arrows
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
    set_seed(SEED)

    # ── Model ───────────────────────────────────────────────────────────
    policy = QAPPolicy()
    critic = CVRPCritic()
    env = CVRPEnv(num_loc=GRAPH_SIZE, device=str(device))
    generator = CVRPGenerator(GRAPH_SIZE, CAPACITY)

    n_actor = sum(p.numel() for p in policy.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())

    # ── Startup banner ──────────────────────────────────────────────────
    print("=" * 72)
    print(f"  QAP-DRL Training — CVRP-{GRAPH_SIZE}")
    print("=" * 72)
    print(f"  Graph size     : {GRAPH_SIZE}")
    print(f"  Capacity       : {CAPACITY}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Epochs         : {N_EPOCHS}")
    print(f"  Epoch size     : {EPOCH_SIZE:,} instances")
    print(f"  Batches/epoch  : {BATCHES_PER_EPOCH}")
    print(f"  Learning rate  : {LR}")
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device         : {device}  ({gpu}, {vram:.1f} GB VRAM)")
    else:
        print(f"  Device         : {device}")
    print(f"  Parameters     : {n_actor} (actor) + {n_critic} (critic)"
          f" = {n_actor + n_critic} total")
    print(f"  Output dir     : {OUTPUT_DIR}")
    print("=" * 72)

    # ── Trainer ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer = PPOTrainer(
        policy=policy, critic=critic, env=env, generator=generator,
        lr=LR, batch_size=BATCH_SIZE, device=str(device), log_dir=OUTPUT_DIR,
    )

    # ── Validation set ──────────────────────────────────────────────────
    val_coords, val_demands, val_cap = load_dataset(VAL_PATH, device=str(device))
    val_instances = (
        val_coords[:VAL_EVAL_SIZE],
        val_demands[:VAL_EVAL_SIZE],
        val_cap,
    )

    # ── Live chart setup ────────────────────────────────────────────────
    plt.ion()
    fig_live, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    fig_live.suptitle(f"QAP-DRL Training — CVRP-{GRAPH_SIZE}", fontsize=14)
    epochs_hist, tour_hist, loss_hist, ent_hist = [], [], [], []

    # ── Training loop ───────────────────────────────────────────────────
    best_tour = float("inf")
    best_epoch = 0

    header = (
        f"{'Epoch':>5} | {'Tour Length':>11} | {'vs LKH3':>9} | "
        f"{'Feasibility':>11} | {'PPO Loss':>9} | {'Entropy':>8} | "
        f"{'VRAM MB':>7} | {'Time':>7}"
    )
    sep = "-" * len(header)
    tqdm.write(f"\n{header}")
    tqdm.write(sep)

    epoch_bar = tqdm(range(1, N_EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rewards, losses_list, feas_list = [], [], []

        batch_bar = tqdm(
            range(BATCHES_PER_EPOCH), desc=f"Epoch {epoch:3d}",
            unit="batch", leave=False,
        )
        for _ in batch_bar:
            mean_rew = trainer.collect_rollout()
            losses = trainer.update()

            with torch.no_grad():
                feas = check_feasibility(
                    trainer._last_instance["demands"],
                    CAPACITY, trainer._last_actions,
                )

            rewards.append(-mean_rew)            # tour length (positive)
            losses_list.append(losses)
            feas_list.append(feas)

            batch_bar.set_postfix(
                ppo_loss=f"{losses['total_loss']:.4f}",
                entropy=f"{losses['entropy']:.4f}",
                feasibility=f"{feas * 100:.1f}%",
            )

        # ── Epoch statistics ────────────────────────────────────────────
        avg_loss = sum(l["total_loss"] for l in losses_list) / len(losses_list)
        avg_ent = sum(l["entropy"] for l in losses_list) / len(losses_list)
        avg_feas = sum(feas_list) / len(feas_list)
        elapsed = time.time() - t0

        vram = (torch.cuda.memory_allocated() / 1e6
                if torch.cuda.is_available() else 0)

        # ── Validation (greedy) ─────────────────────────────────────────
        val_tour = evaluate(trainer.policy, val_instances, device, greedy=True)
        val_gap = 100.0 * (val_tour - LKH3_REF) / LKH3_REF

        # ── Log metrics ────────────────────────────────────────────────
        trainer.logger.log_scalars({
            "val_tour": val_tour, "val_gap": val_gap,
            "ppo_loss": avg_loss, "entropy": avg_ent,
            "feasibility": avg_feas, "vram_mb": vram,
        }, step=epoch)

        # ── Checkpoint every epoch ──────────────────────────────────────
        torch.save({
            "iteration": epoch,
            "policy_state": trainer.policy.state_dict(),
            "critic_state": trainer.critic.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "metrics": {"val_tour": val_tour, "val_gap": val_gap},
        }, os.path.join(OUTPUT_DIR, f"epoch_{epoch:03d}.pt"))

        # ── Best model ──────────────────────────────────────────────────
        is_best = val_tour < best_tour
        if is_best:
            best_tour = val_tour
            best_epoch = epoch
            torch.save(trainer.policy.state_dict(),
                       os.path.join(OUTPUT_DIR, "best_model.pt"))

        epoch_bar.set_postfix(
            best=f"{best_tour:.4f}", current=f"{val_tour:.4f}",
        )

        # ── Summary line ────────────────────────────────────────────────
        marker = " *" if is_best else ""
        tqdm.write(
            f"{epoch:5d} | {val_tour:11.4f} | {val_gap:+8.2f}% | "
            f"{avg_feas * 100:10.2f}% | {avg_loss:9.4f} | {avg_ent:8.4f} | "
            f"{vram:7.0f} | {elapsed:6.1f}s{marker}"
        )

        # ── Update live chart ───────────────────────────────────────────
        epochs_hist.append(epoch)
        tour_hist.append(val_tour)
        loss_hist.append(avg_loss)
        ent_hist.append(avg_ent)

        ax1.cla()
        ax1.plot(epochs_hist, tour_hist, "b-", linewidth=1.5, label="Model")
        ax1.axhline(y=LKH3_REF, color="r", linestyle="--", linewidth=1,
                    label=f"LKH3 ({LKH3_REF})")
        ax1.fill_between(epochs_hist, tour_hist, LKH3_REF,
                         alpha=0.15, color="red")
        ax1.set_title("Tour Length", fontsize=14)
        ax1.set_ylabel("Avg Distance", fontsize=12)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.legend(fontsize=10)

        ax2.cla()
        ax2.plot(epochs_hist, loss_hist, color="orange", linewidth=1.5)
        ax2.set_title("PPO Loss", fontsize=14)
        ax2.set_xlabel("Epoch", fontsize=12)

        ax3.cla()
        ax3.plot(epochs_hist, ent_hist, color="green", linewidth=1.5)
        ax3.set_title("Policy Entropy", fontsize=14)
        ax3.set_xlabel("Epoch", fontsize=12)

        fig_live.tight_layout(rect=[0, 0, 1, 0.93])
        plt.pause(0.1)

    trainer.logger.close()

    # ── Save training curves ────────────────────────────────────────────
    curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    fig_live.savefig(curves_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {curves_path}")

    # ── Route map using best model on validation set ────────────────────
    best_policy = QAPPolicy()
    best_policy.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=device)
    )
    best_policy.to(device)
    best_policy.eval()

    n_eval = min(256, val_coords.size(0))
    bc = val_coords[:n_eval].to(device)
    bd = val_demands[:n_eval].to(device)

    route_env = CVRPEnv(num_loc=GRAPH_SIZE, device=str(device))
    state = route_env.reset({
        "coords": bc, "demands": bd,
        "capacity": torch.full((n_eval,), float(val_cap), device=device),
    })
    with torch.no_grad():
        actions, _, _ = best_policy(state, route_env, deterministic=True)

    T_act = actions.shape[1]
    idx = actions.unsqueeze(-1).expand(n_eval, T_act, 2)
    route_pts = bc.gather(1, idx)
    depot = bc[:, 0:1, :]
    full = torch.cat([depot, route_pts, depot], dim=1)
    dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)

    best_i = dists.argmin().item()
    best_dist = dists[best_i].item()

    plot_route_map(
        bc[best_i].cpu().numpy(),
        actions[best_i].cpu().tolist(),
        GRAPH_SIZE, best_dist,
        f"Best Route — CVRP-{GRAPH_SIZE} | Tour Length: {best_dist:.2f}",
        os.path.join(OUTPUT_DIR, "best_route.png"),
    )

    # ── Final summary ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  Training Complete — CVRP-{GRAPH_SIZE}")
    print("=" * 72)
    print(f"  Best epoch       : {best_epoch}")
    print(f"  Best tour length : {best_tour:.4f}")
    print(f"  vs LKH3 gap      : {100 * (best_tour - LKH3_REF) / LKH3_REF:+.2f}%")
    print(f"  Model saved      : {os.path.join(OUTPUT_DIR, 'best_model.pt')}")
    print("=" * 72)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
