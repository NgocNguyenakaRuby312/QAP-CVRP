"""
encoder/amplitude_projection.py
================================
Step 3 — Amplitude Projection.

Projects each 6-D feature vector onto the 2-D unit circle:

    [αᵢ, βᵢ]ᵀ = Normalize(W·xᵢ + b)     s.t.  αᵢ² + βᵢ² = 1        # Eq §3.X.4

Every node ends up on the unit circle in ℝ², which is the shared
representation space used by the rotation module (Step 4) and
the hybrid scoring decoder (Step 6).

Change 3 (§3.3.1, May 2026):
    input_dim raised 5 → 6 to accept the new dist(i, vₜ) dynamic feature.
    W ∈ ℝ^{2×6}, b ∈ ℝ^2 → 14 params total (was 12, +2 from Change 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmplitudeProjection(nn.Module):
    """
    Linear projection followed by L2 normalisation.

    Args:
        input_dim:  feature dimension  (default 6 — Change 3: was 5)
        output_dim: amplitude dimension (default 2)
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 2):       # Change 3: default 6
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)      # W ∈ ℝ^{2×6}, b ∈ ℝ^2
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N+1, 6]  (Change 3: was 5)

        Returns:
            psi: [B, N+1, 2]  unit-norm amplitude vectors  ‖ψᵢ‖ = 1
        """
        z   = self.linear(features)                                    # [B, N+1, 2]
        psi = F.normalize(z, p=2, dim=-1, eps=1e-6)                   # unit-norm
        return psi

    def angles(self, psi: torch.Tensor) -> torch.Tensor:
        """Recover polar angles φᵢ = atan2(βᵢ, αᵢ) ∈ [-π, π]."""
        return torch.atan2(psi[..., 1], psi[..., 0])
