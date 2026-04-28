"""
encoder/rotation.py
===================
Step 4 — Apply per-node rotation R(θ)·ψ.

Phase 2: 4D rotation via 6 Givens rotations on S³.
    SO(4) has 6 independent planes: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
    Each plane gets one angle θₖ from the RotationMLP.
    The composite rotation R = G₆·G₅·G₄·G₃·G₂·G₁ is an isometry on S³.

Unit norm is preserved: ‖ψ'ᵢ‖ = ‖ψᵢ‖ = 1.
"""

import torch
import torch.nn as nn

from .rotation_mlp import RotationMLP

# The 6 Givens planes for SO(4), in fixed application order
GIVENS_PLANES_4D = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def apply_givens_4d(psi: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """
    Apply 6 Givens rotations to 4D amplitude vectors.                  # Eq: ψ' = R(Θ)·ψ

    Each Givens rotation mixes two dimensions (i,j) by angle θₖ:
        psi[..., i] =  cos(θₖ)·psi[..., i] + sin(θₖ)·psi[..., j]
        psi[..., j] = -sin(θₖ)·psi[..., i] + cos(θₖ)·psi[..., j]

    Args:
        psi:    [B, N+1, 4]  unit-norm amplitude vectors on S³
        thetas: [B, N+1, 6]  one angle per Givens plane

    Returns:
        psi_prime: [B, N+1, 4]  rotated (unit norm preserved)
    """
    out = psi.clone()                                                  # [B, N+1, 4]
    for k, (i, j) in enumerate(GIVENS_PLANES_4D):
        theta_k = thetas[..., k]                                       # [B, N+1]
        c = torch.cos(theta_k)                                        # [B, N+1]
        s = torch.sin(theta_k)                                        # [B, N+1]
        oi = out[..., i].clone()                                       # [B, N+1]
        oj = out[..., j].clone()                                       # [B, N+1]
        out[..., i] =  c * oi + s * oj                                 # Givens rotation
        out[..., j] = -s * oi + c * oj                                 # norm preserved
    return out                                                         # [B, N+1, 4]


def rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """
    Build per-node 2×2 rotation matrices (kept for backward compat).   # Eq: R(θ)

    Args:
        theta: [B, N+1]

    Returns:
        R: [B, N+1, 2, 2]
    """
    c = torch.cos(theta)                                               # [B, N+1]
    s = torch.sin(theta)                                               # [B, N+1]
    R = torch.stack([
        torch.stack([ c, -s], dim=-1),
        torch.stack([ s,  c], dim=-1),
    ], dim=-2)                                                         # [B, N+1, 2, 2]
    return R


def apply_rotation(psi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rotate amplitude vectors. Dispatches to 2D or 4D based on shape.

    Args:
        psi:   [B, N+1, D]  unit-norm amplitude vectors (D=2 or D=4)
        theta: [B, N+1] for 2D, [B, N+1, 6] for 4D

    Returns:
        psi_prime: [B, N+1, D]  rotated (unit norm preserved)
    """
    amp_dim = psi.shape[-1]
    if amp_dim == 4:
        return apply_givens_4d(psi, theta)                             # 4D: 6 Givens planes
    else:
        R = rotation_matrix_2d(theta)                                  # 2D: single R(θ)
        return torch.einsum("bnij,bnj->bni", R, psi)                  # [B, N+1, 2]


class PerNodeRotation(nn.Module):
    """
    Combines RotationMLP + rotation application.

    Args:
        input_dim:  feature dimension (default 5)
        hidden_dim: MLP hidden width  (default 32)
        amp_dim:    amplitude dimension (2 or 4)
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, amp_dim: int = 4):
        super().__init__()
        self.amp_dim = amp_dim
        n_angles = 6 if amp_dim == 4 else 1
        self.mlp = RotationMLP(input_dim, hidden_dim, n_angles=n_angles)

    def forward(self, features: torch.Tensor, psi: torch.Tensor):
        """
        Args:
            features: [B, N+1, 5]
            psi:      [B, N+1, D]  unit-norm amplitude vectors

        Returns:
            psi_prime: [B, N+1, D]  rotated (unit norm preserved)
            theta:     [B, N+1] or [B, N+1, 6]
        """
        theta     = self.mlp(features)                                 # [B,N+1] or [B,N+1,6]
        psi_prime = apply_rotation(psi, theta)                         # [B, N+1, D]
        return psi_prime, theta
