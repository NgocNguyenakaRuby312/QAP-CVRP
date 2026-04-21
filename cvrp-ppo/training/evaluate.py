"""
training/evaluate.py
=====================
Greedy and augmented rollout evaluation.

    evaluate(model, instances, device, greedy=True) → mean_tour_length: float
    evaluate_augmented(model, instances, device, n_samples=8) → mean_best_tour: float

Augmentation strategy: coordinate augmentation + greedy decoding.
    8 standard geometric transforms (rotations + reflections) of the unit square.
    Greedy decoding on each transformed instance.
    Tour length computed on original coords (distance-invariant).
    torch.minimum across 8 greedy runs → valid diversity from different orientations.

AUGMENTATION TRANSFORMS (8 standard for unit square [0,1]^2):
    T0: identity           (x,  y)
    T1: rotate 90°         (1-y, x)
    T2: rotate 180°        (1-x, 1-y)
    T3: rotate 270°        (y,  1-x)
    T4: reflect x          (1-x, y)
    T5: reflect y          (x,  1-y)
    T6: reflect main diag  (y,  x)
    T7: reflect anti diag  (1-y, 1-x)

    All 8 transforms map [0,1]^2 → [0,1]^2 and preserve Euclidean distances.
    Tour length is invariant under all 8.
"""

import torch
from typing import Tuple


# ── 8 standard coordinate transforms for unit square [0,1]^2 ──────────────
# Each is a function: coords [B, N+1, 2] → coords [B, N+1, 2]
# Distances are preserved (isometric transforms).

def _aug_transforms():
    """Return list of 8 coordinate transform lambdas."""
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


@torch.no_grad()
def evaluate_augmented(
    model,
    instances: Tuple[torch.Tensor, torch.Tensor, int],
    device: torch.device,
    n_samples: int = 8,
) -> float:
    """
    Augmented evaluation: coordinate augmentation + greedy decoding.

    For each of n_samples transforms:
        1. Apply a geometric transform to instance coordinates
        2. Run GREEDY decoding (deterministic)
        3. Compute tour length on original coords (distance-invariant)
    Take the per-instance minimum across all n_samples transforms.
    Return the mean of per-instance best tour lengths.

    8 isometric transforms of the unit square are used. Each creates a
    geometrically equivalent but differently-oriented routing problem,
    providing genuine diversity for the minimum search.

    Args:
        model:     QAPPolicy (already on device)
        instances: tuple of (coords [B,N+1,2], demands [B,N+1], capacity int)
        device:    torch device
        n_samples: number of augmented views, 1–8 (default 8)
                   1 = plain greedy, no augmentation overhead

    Returns:
        mean_best_tour: float — mean of per-instance best tours (lower = better)
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

    transforms = _aug_transforms()
    # Use at most 8 transforms (the full symmetry group of the square)
    n_aug = min(n_samples, len(transforms))

    best_dists = None

    for i in range(n_aug):
        # ── Apply coordinate transform ────────────────────────────────
        # Demands and capacity are unchanged — only coordinates rotate/reflect.
        # Distance is invariant to isometric transforms so tour_length is valid.
        aug_coords = transforms[i](coords_dev)                         # [B, N+1, 2]

        env = CVRPEnv(num_loc=N, device=str(device))
        state = env.reset({
            "coords":   aug_coords,
            "demands":  demands_dev,
            "capacity": cap_tensor,
        })

        # ── GREEDY decode on augmented instance ──────────────────────
        # deterministic=True ensures encoder sees a consistent v_t sequence
        # throughout the rollout — required for Change 3 validity.
        actions, _, _ = model(state, env, deterministic=True)

        # ── Tour length on ORIGINAL coords (distance-invariant) ───────
        # We compute the tour length on the ORIGINAL (non-transformed) coords
        # because the node indices (action sequence) are the same regardless
        # of the coordinate transform — only the routing order differs.
        T_steps = actions.shape[1]
        idx     = actions.unsqueeze(-1).expand(B, T_steps, 2)
        route   = coords_dev.gather(1, idx)                            # [B, T, 2]
        depot   = coords_dev[:, 0:1, :]                                # [B, 1, 2]
        full    = torch.cat([depot, route, depot], dim=1)              # [B, T+2, 2]
        dists   = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)  # [B]

        if best_dists is None:
            best_dists = dists
        else:
            best_dists = torch.minimum(best_dists, dists)              # [B] per-inst min

    return best_dists.mean().item()
