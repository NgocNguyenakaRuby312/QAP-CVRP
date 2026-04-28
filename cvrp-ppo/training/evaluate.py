"""
training/evaluate.py
=====================
Greedy and augmented rollout evaluation.

    evaluate(model, instances, device, greedy=True) -> mean_tour_length: float
    evaluate_augmented(model, instances, device, n_samples=16) -> mean_best_tour: float

Augmentation strategy (Phase 2+):
    16 geometric transforms (8 standard + 8 with random angle offsets)
    For each transform: 1 greedy + K stochastic samples at low temperature
    torch.minimum across all attempts -> per-instance best tour
"""

import torch
from typing import Tuple
import math


# -- 8 standard coordinate transforms for unit square [0,1]^2 ---------------
def _aug_transforms():
    """Return list of 8 standard isometric transforms."""
    return [
        lambda c: c,                                                    # T0: identity
        lambda c: torch.stack([1-c[...,1], c[...,0]],    dim=-1),     # T1: rot 90
        lambda c: torch.stack([1-c[...,0], 1-c[...,1]],  dim=-1),     # T2: rot 180
        lambda c: torch.stack([c[...,1],   1-c[...,0]],  dim=-1),     # T3: rot 270
        lambda c: torch.stack([1-c[...,0], c[...,1]],    dim=-1),     # T4: flip x
        lambda c: torch.stack([c[...,0],   1-c[...,1]],  dim=-1),     # T5: flip y
        lambda c: torch.stack([c[...,1],   c[...,0]],    dim=-1),     # T6: diag
        lambda c: torch.stack([1-c[...,1], 1-c[...,0]],  dim=-1),     # T7: anti-diag
    ]


def _continuous_rotation(coords: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate coords around (0.5, 0.5) by angle_deg degrees, then clamp to [0,1]."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    centered = coords - 0.5                                            # center at origin
    rotated = torch.stack([
        c * centered[..., 0] - s * centered[..., 1],
        s * centered[..., 0] + c * centered[..., 1],
    ], dim=-1)
    return (rotated + 0.5).clamp(0.0, 1.0)                            # back to [0,1]


def _extended_transforms():
    """Return list of 16 transforms: 8 standard + 8 with 45-degree offsets."""
    base = _aug_transforms()
    extra = [
        lambda c: _continuous_rotation(c, 45),
        lambda c: _continuous_rotation(c, 135),
        lambda c: _continuous_rotation(c, 225),
        lambda c: _continuous_rotation(c, 315),
        lambda c: _continuous_rotation(c, 22.5),
        lambda c: _continuous_rotation(c, 67.5),
        lambda c: _continuous_rotation(c, 112.5),
        lambda c: _continuous_rotation(c, 157.5),
    ]
    return base + extra


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

    coords_dev  = coords.to(device)
    demands_dev = demands.to(device)
    cap_tensor  = torch.full((B,), float(capacity), device=device)

    env = CVRPEnv(num_loc=N, device=str(device))
    state = env.reset({
        "coords":   coords_dev,
        "demands":  demands_dev,
        "capacity": cap_tensor,
    })

    actions, _, _ = model(state, env, deterministic=greedy)

    T     = actions.shape[1]
    idx   = actions.unsqueeze(-1).expand(B, T, 2)
    route = coords_dev.gather(1, idx)
    depot = coords_dev[:, 0:1, :]
    full  = torch.cat([depot, route, depot], dim=1)
    dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)    # [B]

    return dists.mean().item()


def _compute_tour_length(coords_dev, actions):
    """Helper: compute tour length on original coords given action sequence."""
    B, T = actions.shape
    idx   = actions.unsqueeze(-1).expand(B, T, 2)
    route = coords_dev.gather(1, idx)                                  # [B, T, 2]
    depot = coords_dev[:, 0:1, :]                                      # [B, 1, 2]
    full  = torch.cat([depot, route, depot], dim=1)                    # [B, T+2, 2]
    return (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)      # [B]


@torch.no_grad()
def evaluate_augmented(
    model,
    instances: Tuple[torch.Tensor, torch.Tensor, int],
    device: torch.device,
    n_samples: int = 16,
    n_stochastic: int = 3,
    temperature: float = 1.2,
) -> float:
    """
    Augmented evaluation: coordinate augmentation + greedy + stochastic sampling.

    For each of n_samples transforms:
        1. Apply geometric transform to coordinates
        2. Run 1 GREEDY decode (deterministic)
        3. Run n_stochastic STOCHASTIC decodes at given temperature
        4. Compute tour length on ORIGINAL coords
    Take per-instance minimum across all attempts.

    Total attempts = n_samples * (1 + n_stochastic)
    Default: 16 * (1 + 3) = 64 attempts per instance

    Args:
        model:         QAPPolicy (already on device)
        instances:     tuple of (coords, demands, capacity)
        device:        torch device
        n_samples:     number of coordinate transforms (default 16)
        n_stochastic:  stochastic samples per transform (default 3, 0=greedy only)
        temperature:   sampling temperature (default 1.2, >1 = more exploration)

    Returns:
        mean_best_tour: float (lower = better)
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

    if n_samples <= 8:
        transforms = _aug_transforms()[:n_samples]
    else:
        transforms = _extended_transforms()[:n_samples]

    best_dists = None

    for i in range(len(transforms)):
        aug_coords = transforms[i](coords_dev)                         # [B, N+1, 2]

        # -- Greedy decode (always) ------------------------------------
        env = CVRPEnv(num_loc=N, device=str(device))
        state = env.reset({
            "coords":   aug_coords,
            "demands":  demands_dev,
            "capacity": cap_tensor,
        })
        actions, _, _ = model(state, env, deterministic=True)
        dists = _compute_tour_length(coords_dev, actions)              # [B]
        best_dists = dists if best_dists is None else torch.minimum(best_dists, dists)

        # -- Stochastic decodes (low temperature) ----------------------
        for _ in range(n_stochastic):
            env_s = CVRPEnv(num_loc=N, device=str(device))
            state_s = env_s.reset({
                "coords":   aug_coords,
                "demands":  demands_dev,
                "capacity": cap_tensor,
            })

            # Run decoder manually with temperature-scaled logits
            psi_prime, _, knn_indices = model.encoder(state_s)
            n_customers = psi_prime.shape[1] - 1
            all_actions = []
            st = state_s
            max_steps = 3 * n_customers + 1

            for step in range(max_steps):
                if st["done"].all():
                    break
                log_probs, _ = model.decoder(
                    st, psi_prime, knn_indices, step, n_customers
                )
                # Temperature scaling: divide logits by temperature before sampling
                scaled_logits = log_probs / temperature
                dist = torch.distributions.Categorical(logits=scaled_logits)
                action = dist.sample()
                all_actions.append(action)
                st = env_s.step(st, action)

            if all_actions:
                actions_s = torch.stack(all_actions, dim=1)
                dists_s = _compute_tour_length(coords_dev, actions_s)
                best_dists = torch.minimum(best_dists, dists_s)

    return best_dists.mean().item()
