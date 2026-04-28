"""
encoder/rotation_mlp.py
========================
MLP that predicts per-node rotation angles.

Phase 2 (4D):
    θᵢ = MLP(xᵢ)   MLP: 5 → 32 → 6 (one angle per Givens plane in SO(4))
    Parameters: 5×32 + 32 + 32×6 + 6 = 230

Phase 1 (2D, backward compat):
    θᵢ = MLP(xᵢ)   MLP: 5 → 32 → 1
    Parameters: 5×32 + 32 + 32×1 + 1 = 225
"""

import torch
import torch.nn as nn


class RotationMLP(nn.Module):
    """
    MLP: ℝ⁵ → ℝⁿ  predicts rotation angles.

    Args:
        input_dim:  feature dimension (default 5)
        hidden_dim: hidden layer width (default 32)
        n_angles:   number of output angles (1 for 2D, 6 for 4D SO(4))
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, n_angles: int = 6):
        super().__init__()
        self.n_angles = n_angles
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),                          # 5 → 32
            nn.Tanh(),                                                 # tanh activation
            nn.Linear(hidden_dim, n_angles),                           # 32 → n_angles
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N+1, 5]

        Returns:
            theta: [B, N+1] if n_angles==1, else [B, N+1, n_angles]
        """
        out = self.net(features)                                       # [B, N+1, n_angles]
        out = torch.clamp(out, -10, 10)                                # P5: prevent NaN
        if self.n_angles == 1:
            return out.squeeze(-1)                                     # [B, N+1]  backward compat
        return out                                                     # [B, N+1, 6]
