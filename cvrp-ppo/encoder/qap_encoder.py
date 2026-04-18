"""
encoder/qap_encoder.py
=======================
Combines: AmplitudeProjection + RotationMLP + apply_rotation.

    features [B, N+1, 5] → psi_prime [B, N+1, 2]   (unit norm)

Also exports FullEncoder — a top-level wrapper that chains
FeatureBuilder → QAPEncoder → kNN lookup.  Used by QAPPolicy.
"""

import sys
import os
import torch
import torch.nn as nn

from .feature_constructor  import FeatureBuilder
from .amplitude_projection import AmplitudeProjection
from .rotation_mlp         import RotationMLP
from .rotation             import apply_rotation

# utils/ lives one level up from encoder/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.knn import compute_knn


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


class FullEncoder(nn.Module):
    """
    Top-level encoder wrapper: state dict → (psi_prime, features, knn_indices).

    Chains FeatureBuilder + QAPEncoder + spatial kNN.
    Used by QAPPolicy.forward() and the training loop.

    Args:
        input_dim:  node feature size   (default 5)
        amp_dim:    amplitude space dim (default 2)
        hidden_dim: rotation MLP width  (default 16)
        knn_k:      kNN neighbourhood   (default 5)
    """

    def __init__(self, input_dim: int = 5, amp_dim: int = 2,
                 hidden_dim: int = 16, knn_k: int = 5):
        super().__init__()
        self.feature_builder = FeatureBuilder()
        self.qap_encoder     = QAPEncoder(input_dim, amp_dim, hidden_dim)
        self.knn_k           = knn_k

    def forward(self, state: dict):
        """
        Args:
            state: dict with coords [B,N+1,2], demands [B,N+1], capacity

        Returns:
            psi_prime:   [B, N+1, 2]  unit-norm rotated embeddings
            features:    [B, N+1, 5]  raw 5D features
            knn_indices: [B, N+1, k]  spatial kNN (no self-loops)
        """
        features    = self.feature_builder(state)                      # [B, N+1, 5]
        psi_prime   = self.qap_encoder(features)                       # [B, N+1, 2]
        knn_indices = compute_knn(state["coords"], self.knn_k)         # [B, N+1, k]
        return psi_prime, features, knn_indices
