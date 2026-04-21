"""
encoder/rotation_mlp.py
========================
MLP that predicts per-node rotation angle θᵢ.

    θᵢ = MLP(xᵢ)      MLP: 6 → 16 → 1, tanh activation               # Eq §3.X.5

Change 3 (§3.3.1, May 2026):
    input_dim raised 5 → 6 to accept the new dist(i, vₜ) dynamic feature.
    First Linear grows from 5×16 (80 params) to 6×16 (96 params): +16 params.
    The sixth input column teaches the rotation MLP to assign larger rotation
    angles to nodes that are far from the vehicle's current position, making
    the amplitude geometry proximity-sensitive.

Architecture: Linear(6→16) → Tanh → Linear(16→1)
Parameters:   6×16 + 16 + 16×1 + 1 = 129  (was 113, +16 from Change 3)
"""

import torch
import torch.nn as nn


class RotationMLP(nn.Module):
    """
    MLP: ℝ⁶ → ℝ¹  predicts rotation angle θᵢ.

    Args:
        input_dim:  feature dimension (default 6 — Change 3: was 5)
        hidden_dim: hidden layer width (default 16)
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 16):  # Change 3: default 6
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),                          # 6 → 16  (Change 3)
            nn.Tanh(),                                                 # tanh activation
            nn.Linear(hidden_dim, 1),                                  # 16 → 1
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N+1, 6]  (Change 3: was 5)

        Returns:
            theta: [B, N+1]  rotation angles
        """
        theta = self.net(features).squeeze(-1)                         # [B, N+1]
        theta = torch.clamp(theta, -10, 10)                            # P5: prevent NaN
        return theta
