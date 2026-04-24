"""
encoder/amplitude_projection.py
================================
Step 3 — Amplitude Projection.

Projects each 5-D feature vector onto the unit hypersphere:

    ψᵢ = Normalize(W·xᵢ + b)     s.t.  ‖ψᵢ‖ = 1                      # Eq §3.X.4

Phase 2 (4D): W ∈ ℝ^{4×5}, b ∈ ℝ^4 → 24 params.  ψ ∈ S³.
Phase 1 (2D): W ∈ ℝ^{2×5}, b ∈ ℝ^2 → 12 params.  ψ ∈ S¹.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmplitudeProjection(nn.Module):
    """
    Linear projection followed by L2 normalisation.

    Args:
        input_dim:  feature dimension  (default 5)
        output_dim: amplitude dimension (default 4 for Phase 2)
    """

    def __init__(self, input_dim: int = 5, output_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)      # W ∈ ℝ^{D×5}
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N+1, 5]

        Returns:
            psi: [B, N+1, D]  unit-norm amplitude vectors  ‖ψᵢ‖ = 1
        """
        z   = self.linear(features)                                    # [B, N+1, D]
        psi = F.normalize(z, p=2, dim=-1, eps=1e-6)                   # unit-norm on S^(D-1)
        return psi
