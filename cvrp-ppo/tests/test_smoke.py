"""
tests/test_smoke.py — End-to-end: B=2, N=5, 50 training steps, reward must improve.
Run: python tests/test_smoke.py   OR   pytest tests/test_smoke.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.optim import Adam
from models.qap_policy import QAPPolicy
from environment.cvrp_env import CVRPEnv
from utils.data_generator import generate_instances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B, N, CAP = 2, 5, 15


def _make_instances():
    torch.manual_seed(42)
    coords, demands, cap = generate_instances(B, N, capacity=CAP, device=str(device))
    return coords, demands, cap


def _rollout_and_train(model, optimizer, env, coords, demands, cap):
    """One REINFORCE step. Returns mean reward (negative distance)."""
    model.train()
    state = env.reset({
        "coords":   coords,
        "demands":  demands,
        "capacity": torch.full((B,), float(cap), device=device),
    })
    actions, log_probs_t, sum_lp = model(state, env, deterministic=False)

    # Tour length
    T = actions.shape[1]
    idx   = actions.unsqueeze(-1).expand(B, T, 2)                      # [B, T, 2]
    route = coords.gather(1, idx)                                      # [B, T, 2]
    depot = coords[:, 0:1, :]                                          # [B, 1, 2]
    full  = torch.cat([depot, route, depot], dim=1)                    # [B, T+2, 2]
    dist  = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)    # [B]

    reward = -dist                                                      # [B] higher = better
    baseline = reward.mean()
    advantage = reward - baseline
    loss = -(sum_lp * advantage.detach()).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return reward.mean().item()


def test_smoke():
    print(f"Smoke test running on: {device}")

    model = QAPPolicy(feature_dim=5, hidden_dim=16, knn_k=min(5, N),
                       lambda_init=0.1).to(device)
    optimizer = Adam(model.parameters(), lr=5e-3)
    env = CVRPEnv(num_loc=N, device=str(device))
    coords, demands, cap = _make_instances()

    # Collect initial reward (sampling, not greedy — more sensitive to policy changes)
    rewards = []
    for _ in range(5):
        r = _rollout_and_train(model, optimizer, env, coords, demands, cap)
        rewards.append(r)
    r_start = sum(rewards) / len(rewards)

    # Train more steps
    for step in range(50):
        _rollout_and_train(model, optimizer, env, coords, demands, cap)

    # Collect final reward
    rewards = []
    for _ in range(5):
        r = _rollout_and_train(model, optimizer, env, coords, demands, cap)
        rewards.append(r)
    r_end = sum(rewards) / len(rewards)

    print(f"  start={r_start:.4f}  end={r_end:.4f}")
    assert r_end > r_start, f"No improvement: {r_start:.4f} -> {r_end:.4f}"
    print(f"PASSED: {r_start:.4f} -> {r_end:.4f} on {device}")

    if device.type == "cuda":
        print(f"VRAM used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")


if __name__ == "__main__":
    test_smoke()
