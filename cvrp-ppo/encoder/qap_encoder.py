"""
encoder/qap_encoder.py
=======================
Combines: AmplitudeProjection + RotationMLP + apply_rotation.

    features [B, N+1, 5] → psi_prime [B, N+1, 2]   (unit norm)
"""

import torch
import torch.nn as nn

from .amplitude_projection import AmplitudeProjection
from .rotation_mlp         import RotationMLP
from .rotation             import apply_rotation


class QAPEncoder(nn.Module):
    """
    Full encoder: projection → rotation.

    Args:
        input_dim:  feature dimension (default 5)
        amp_dim:    amplitude space dimension (default 2)
        hidden_dim: rotation MLP hidden width (default 16)
    """

    def __init__(self, input_dim: int = 5, amp_dim: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.amplitude_proj = AmplitudeProjection(input_dim, amp_dim)
        self.rotation_mlp   = RotationMLP(input_dim, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N+1, 5]

        Returns:
            psi_prime: [B, N+1, 2]  unit-norm rotated embeddings
        """
        psi   = self.amplitude_proj(features)                          # [B, N+1, 2]  Eq: ψ=Norm(W·x+b)
        theta = self.rotation_mlp(features)                            # [B, N+1]     Eq: θ=MLP(x)
        psi_prime = apply_rotation(psi, theta)                         # [B, N+1, 2]  Eq: ψ'=R(θ)·ψ
        return psi_prime
