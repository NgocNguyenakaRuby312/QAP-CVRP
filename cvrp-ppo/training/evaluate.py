"""
training/evaluate.py
=====================
Greedy and augmented rollout evaluation.

    evaluate(model, instances, device, greedy=True) → mean_tour_length: float
    evaluate_augmented(model, instances, device, n_samples=8) → mean_best_tour: float

evaluate_augmented:
    Runs n_samples stochastic rollouts per instance, returns the mean of
    the per-instance best tour length across all val instances.
    This is the POMO-style inference augmentation — no retraining needed.
    Typical gain: 2–4% gap reduction vs pure greedy decoding.
"""

import torch
from typing import Tuple


@torch.no_grad()
def evaluate(
    model,
    instances: Tuple[torch.Tensor, torch.Tensor, int],
    device: torch.device,
    greedy: bool = True,
) -> float:
    """
    Evaluate model on a batch of CVRP instances.

    Args:
        model:     QAPPolicy (already on device)
        instances: tuple of (coords [B,N+1,2], demands [B,N+1], capacity int)
        device:    torch device
        greedy:    True = argmax at each step (deterministic)

    Returns:
        mean_tour_length: float (positive distance, lower = better)
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from environment.cvrp_env import CVRPEnv

    model.eval()

    coords, demands, capacity = instances
    B = coords.size(0)
    N = coords.size(1) - 1

    env = CVRPEnv(num_loc=N, device=str(device))
    state = env.reset({
        "coords":   coords.to(device),
        "demands":  demands.to(device),
        "capacity": torch.full((B,), float(capacity), device=device),
    })

    actions, _, _ = model(state, env, deterministic=greedy)

    T     = actions.shape[1]
    idx   = actions.unsqueeze(-1).expand(B, T, 2)
    route = coords.to(device).gather(1, idx)
    depot = coords[:, 0:1, :].to(device)
    full  = torch.cat([depot, route, depot], dim=1)
    dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)    # [B]

    return dists.mean().item()


@torch.no_grad()
def evaluate_augmented(
    model,
    instances: Tuple[torch.Tensor, torch.Tensor, int],
    device: torch.device,
    n_samples: int = 8,
) -> float:
    """
    Augmented evaluation: sample n_samples stochastic rollouts per instance,
    keep the best (shortest) tour per instance, return the mean best tour.

    This is inference-only augmentation — no retraining, no architecture change.
    Equivalent to running the stochastic policy n_samples times and taking the
    best solution found, mimicking POMO's multi-rollout strategy.

    Args:
        model:     QAPPolicy (already on device)
        instances: tuple of (coords [B,N+1,2], demands [B,N+1], capacity int)
        device:    torch device
        n_samples: number of stochastic samples per instance (default 8)

    Returns:
        mean_best_tour: float — mean of per-instance best tours (lower = better)

    Example: with n_samples=8 and B=500 val instances:
        - runs 8 forward passes, each producing 500 tour lengths
        - takes element-wise min across the 8 runs → [500] best tours
        - returns mean of those 500 best tours
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from environment.cvrp_env import CVRPEnv

    model.eval()

    coords, demands, capacity = instances
    B = coords.size(0)
    N = coords.size(1) - 1

    coords_dev  = coords.to(device)
    demands_dev = demands.to(device)
    cap_tensor  = torch.full((B,), float(capacity), device=device)

    # Accumulate best tour length per instance across all samples
    best_dists = None

    for _ in range(n_samples):
        env = CVRPEnv(num_loc=N, device=str(device))
        state = env.reset({
            "coords":   coords_dev,
            "demands":  demands_dev,
            "capacity": cap_tensor,
        })

        # deterministic=False → stochastic sampling
        actions, _, _ = model(state, env, deterministic=False)

        T     = actions.shape[1]
        idx   = actions.unsqueeze(-1).expand(B, T, 2)
        route = coords_dev.gather(1, idx)                              # [B, T, 2]
        depot = coords_dev[:, 0:1, :]                                  # [B, 1, 2]
        full  = torch.cat([depot, route, depot], dim=1)                # [B, T+2, 2]
        dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1) # [B]

        if best_dists is None:
            best_dists = dists
        else:
            best_dists = torch.minimum(best_dists, dists)              # element-wise min

    return best_dists.mean().item()
