"""
encoder/rotation_mlp.py
========================
MLP that predicts per-node rotation angle θᵢ.

    θᵢ = MLP(xᵢ)      MLP: 5 → 16 → 1, tanh activation               # Eq §3.X.5

Architecture: Linear(5→16) → Tanh → Linear(16→1)
Parameters:   5×16 + 16 + 16×1 + 1 = 113
"""

import torch
import torch.nn as nn


class RotationMLP(nn.Module):
    """
    MLP: ℝ⁵ → ℝ¹  predicts rotation angle θᵢ.

    Args:
        input_dim:  feature dimension (default 5)
        hidden_dim: hidden layer width (default 16)
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),                          # 5 → 16
            nn.Tanh(),                                                 # tanh activation
            nn.Linear(hidden_dim, 1),                                  # 16 → 1
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N+1, 5]

        Returns:
            theta: [B, N+1]  rotation angles
        """
        theta = self.net(features).squeeze(-1)                         # [B, N+1]
        theta = torch.clamp(theta, -10, 10)                            # P5: prevent NaN
        return theta
