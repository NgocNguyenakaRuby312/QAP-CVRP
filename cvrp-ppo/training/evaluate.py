"""
training/evaluate.py
=====================
Greedy rollout evaluation.

    evaluate(model, instances, device, greedy=True) → mean_tour_length: float
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
                   tensors already on device
        device:    torch device
        greedy:    True = argmax at each step (no sampling)

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
        "coords":   coords.to(device),                                # [B, N+1, 2]
        "demands":  demands.to(device),                                # [B, N+1]
        "capacity": torch.full((B,), float(capacity), device=device),  # [B]
    })

    actions, _, _ = model(state, env, deterministic=greedy)

    # Tour length: depot → route → depot
    T = actions.shape[1]
    idx   = actions.unsqueeze(-1).expand(B, T, 2)                      # [B, T, 2]
    route = coords.to(device).gather(1, idx)                           # [B, T, 2]
    depot = coords[:, 0:1, :].to(device)                               # [B, 1, 2]
    full  = torch.cat([depot, route, depot], dim=1)                    # [B, T+2, 2]
    dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)    # [B]

    return dists.mean().item()
