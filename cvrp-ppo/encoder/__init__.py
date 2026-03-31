"""
encoder/__init__.py
===================
Re-exports + convenience wrapper that chains feature_builder → qap_encoder → knn.
"""

import torch
import torch.nn as nn

from .feature_constructor  import FeatureBuilder
from .qap_encoder          import QAPEncoder
from .amplitude_projection import AmplitudeProjection
from .rotation_mlp         import RotationMLP
from .rotation             import PerNodeRotation, apply_rotation, rotation_matrix_2d

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.knn import compute_knn


class FullEncoder(nn.Module):
    """
    Convenience wrapper: state dict → (psi_prime, features, knn_indices).

    Chains FeatureBuilder + QAPEncoder + spatial kNN.
    Used by QAPPolicy.forward() and the training loop.
    """

    def __init__(self, input_dim: int = 5, amp_dim: int = 2,
                 hidden_dim: int = 16, knn_k: int = 5):
        super().__init__()
        self.feature_builder = FeatureBuilder()
        self.qap_encoder     = QAPEncoder(input_dim, amp_dim, hidden_dim)
        self.knn_k = knn_k

    def forward(self, state: dict):
        """
        Args:
            state: dict with coords, demands, capacity

        Returns:
            psi_prime:   [B, N+1, 2]
            features:    [B, N+1, 5]
            knn_indices: [B, N+1, k]
        """
        features    = self.feature_builder(state)                      # [B, N+1, 5]
        psi_prime   = self.qap_encoder(features)                       # [B, N+1, 2]
        knn_indices = compute_knn(state["coords"], self.knn_k)         # [B, N+1, k]
        return psi_prime, features, knn_indices


# Backward-compat alias
CVRPEncoder = FullEncoder
