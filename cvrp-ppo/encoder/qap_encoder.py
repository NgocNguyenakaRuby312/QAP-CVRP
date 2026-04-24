"""
encoder/qap_encoder.py
=======================
Combines: FeatureBuilder + AmplitudeProjection + RotationMLP + apply_rotation.

Phase 2 (4D):
    Features [B,N+1,5] → AmplitudeProjection → ψ [B,N+1,4]
    → RotationMLP → θ [B,N+1,6]  (6 Givens planes for SO(4))
    → apply_rotation(ψ,θ) → ψ' [B,N+1,4]  (unit norm preserved on S³)

The encoder is STATIC — called ONCE before the decoding loop.
psi_prime is fixed for all decode steps. kNN is precomputed from spatial coords.
"""

import sys
import os
import torch
import torch.nn as nn

from .feature_constructor  import FeatureBuilder
from .amplitude_projection import AmplitudeProjection
from .rotation_mlp         import RotationMLP
from .rotation             import apply_rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.knn import compute_knn


class QAPEncoder(nn.Module):
    """
    Full encoder: projection → rotation.

    Args:
        input_dim:  feature dimension (default 5)
        amp_dim:    amplitude space dimension (default 4 for Phase 2)
        hidden_dim: rotation MLP hidden width (default 16)
    """

    def __init__(self, input_dim: int = 5, amp_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        n_angles = 6 if amp_dim == 4 else 1                           # SO(4)=6, SO(2)=1
        self.amplitude_proj = AmplitudeProjection(input_dim, amp_dim)  # 5→D
        self.rotation_mlp   = RotationMLP(input_dim, hidden_dim,
                                          n_angles=n_angles)           # 5→16→n_angles

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N+1, 5]

        Returns:
            psi_prime: [B, N+1, D]  unit-norm rotated embeddings
        """
        psi   = self.amplitude_proj(features)                          # [B, N+1, D]
        theta = self.rotation_mlp(features)                            # [B,N+1] or [B,N+1,6]
        psi_prime = apply_rotation(psi, theta)                         # [B, N+1, D]
        return psi_prime


class FullEncoder(nn.Module):
    """
    Top-level encoder wrapper: state dict → (psi_prime, features, knn_indices).

    Args:
        input_dim:  node feature size   (default 5)
        amp_dim:    amplitude space dim (default 4 for Phase 2)
        hidden_dim: rotation MLP width  (default 16)
        knn_k:      kNN neighbourhood   (default 5)
    """

    def __init__(self, input_dim: int = 5, amp_dim: int = 4,
                 hidden_dim: int = 16, knn_k: int = 5):
        super().__init__()
        self.feature_builder = FeatureBuilder()
        self.qap_encoder     = QAPEncoder(input_dim, amp_dim, hidden_dim)
        self.knn_k           = knn_k

    def forward(self, state: dict):
        """
        Returns:
            psi_prime:   [B, N+1, D]  unit-norm rotated embeddings
            features:    [B, N+1, 5]  5D static features
            knn_indices: [B, N+1, k]  spatial kNN (no self-loops)
        """
        features    = self.feature_builder(state)                       # [B, N+1, 5]
        psi_prime   = self.qap_encoder(features)                        # [B, N+1, D]
        knn_indices = compute_knn(state["coords"], self.knn_k)          # [B, N+1, k]
        return psi_prime, features, knn_indices
