"""
encoder/qap_encoder.py
=======================
Combines: AmplitudeProjection + RotationMLP + apply_rotation.

Change 3 (§3.3.1, May 2026):
    input_dim raised 5 → 6 throughout.  FeatureBuilder now outputs [B, N+1, 6]
    (adding dist(i, vₜ) as the 6th feature), and both AmplitudeProjection and
    RotationMLP are initialised with input_dim=6.

    The FullEncoder.forward() signature now accepts an optional
    current_node_coords [B, 2] argument and passes it through to
    FeatureBuilder.  This is required during the autoregressive decoding
    loop so that feature[5] = dist(i, current_node) is computed with the
    actual vehicle position at each step.

    FullEncoder is called in two modes:
        1. One-shot (initial) encode:  no current_node_coords supplied →
           feature[5] falls back to dist(i, depot).  Used to warm-start
           the encoder before the first decoding step (e.g. kNN pre-computation).
        2. Per-step dynamic encode:    current_node_coords supplied each step →
           feature[5] = live distance.  Used inside the decoding loop.

    NOTE: ppo_agent.update() calls self.policy.encoder.qap_encoder(features_mb)
    directly for minibatch re-evaluation.  evaluate_actions() in qap_policy.py
    re-builds features_mb with per-step current_node_coords via the new
    helper method FullEncoder.build_features(state, current_node_coords).

    features return shape: [B, N+1, 6]  (was [B, N+1, 5])
    psi_prime shape:       [B, N+1, 2]  (unchanged)
    knn_indices shape:     [B, N+1, k]  (unchanged)
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Optional

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
        input_dim:  feature dimension (default 6 — Change 3: was 5)
        amp_dim:    amplitude space dimension (default 2)
        hidden_dim: rotation MLP hidden width (default 16)
    """

    def __init__(self, input_dim: int = 6, amp_dim: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.amplitude_proj = AmplitudeProjection(input_dim, amp_dim)  # Change 3: 6→2
        self.rotation_mlp   = RotationMLP(input_dim, hidden_dim)       # Change 3: 6→16→1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N+1, 6]  (Change 3: was 5)

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
        input_dim:  node feature size   (default 6 — Change 3: was 5)
        amp_dim:    amplitude space dim (default 2)
        hidden_dim: rotation MLP width  (default 16)
        knn_k:      kNN neighbourhood   (default 5)
    """

    def __init__(self, input_dim: int = 6, amp_dim: int = 2,  # Change 3: default 6
                 hidden_dim: int = 16, knn_k: int = 5):
        super().__init__()
        self.feature_builder = FeatureBuilder()
        self.qap_encoder     = QAPEncoder(input_dim, amp_dim, hidden_dim)
        self.knn_k           = knn_k

    def build_features(
        self,
        state: dict,
        current_node_coords: Optional[torch.Tensor] = None,  # [B, 2]  Change 3
    ) -> torch.Tensor:
        """
        Build 6D feature tensor with optional dynamic proximity feature.

        Args:
            state:               dict with coords, demands, capacity
            current_node_coords: [B, 2]  vehicle current position for feature[5].
                                  If None, feature[5] = dist(i, depot).

        Returns:
            features: [B, N+1, 6]
        """
        return self.feature_builder(state, current_node_coords)        # [B, N+1, 6]

    def forward(
        self,
        state: dict,
        current_node_coords: Optional[torch.Tensor] = None,  # [B, 2]  Change 3
    ):
        """
        Args:
            state: dict with coords [B,N+1,2], demands [B,N+1], capacity
            current_node_coords: [B, 2]  Change 3 — current vehicle position.
                When None (initial encoding before decoding loop), feature[5]
                falls back to dist(i, depot).

        Returns:
            psi_prime:   [B, N+1, 2]  unit-norm rotated embeddings
            features:    [B, N+1, 6]  6D features (Change 3: was 5D)
            knn_indices: [B, N+1, k]  spatial kNN (no self-loops)
        """
        features    = self.feature_builder(state, current_node_coords)  # [B, N+1, 6]
        psi_prime   = self.qap_encoder(features)                         # [B, N+1, 2]
        knn_indices = compute_knn(state["coords"], self.knn_k)           # [B, N+1, k]
        return psi_prime, features, knn_indices
