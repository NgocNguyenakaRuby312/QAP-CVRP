"""
encoder/rotation.py
===================
Step 4 — Apply per-node rotation R(θ)·ψ.

    ψ'ᵢ = R(θᵢ) · ψᵢ               2×2 rotation matrix               # Eq §3.X.5

Unit norm is preserved: ‖ψ'ᵢ‖ = ‖ψᵢ‖ = 1  (rotation is an isometry).
No re-normalisation needed after rotation.
"""

import torch
import torch.nn as nn

from .rotation_mlp import RotationMLP


def rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """
    Build per-node 2×2 rotation matrices.                              # Eq: R(θ)

    Args:
        theta: [B, N+1]

    Returns:
        R: [B, N+1, 2, 2]
           [[cos θ, −sin θ],
            [sin θ,  cos θ]]
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
    Rotate amplitude vectors by per-node angles.                       # Eq: ψ' = R(θ)·ψ

    Args:
        psi:   [B, N+1, 2]  unit-norm amplitude vectors
        theta: [B, N+1]     rotation angles

    Returns:
        psi_prime: [B, N+1, 2]  rotated (unit norm preserved)
    """
    R = rotation_matrix_2d(theta)                                      # [B, N+1, 2, 2]
    psi_prime = torch.einsum("bnij,bnj->bni", R, psi)                 # [B, N+1, 2]
    return psi_prime


class PerNodeRotation(nn.Module):
    """
    Combines RotationMLP + rotation application.                       # Eq: ψ' = R(θ)·ψ

    Args:
        input_dim:  feature dimension (default 5)
        hidden_dim: MLP hidden width  (default 16)
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.mlp = RotationMLP(input_dim, hidden_dim)

    def forward(self, features: torch.Tensor, psi: torch.Tensor):
        """
        Args:
            features: [B, N+1, 5]  raw node features
            psi:      [B, N+1, 2]  unit-norm amplitude vectors (Step 3)

        Returns:
            psi_prime: [B, N+1, 2]  rotated embeddings (unit norm preserved)
            theta:     [B, N+1]    rotation angles
        """
        theta     = self.mlp(features)                                 # [B, N+1]
        psi_prime = apply_rotation(psi, theta)                         # [B, N+1, 2]
        return psi_prime, theta
