"""
encoder/baseline_encoder.py
============================
Pure DRL baseline encoder — NO quantum-amplitude components.

Used for Ablation Study Tier 2, variant (b):
    "Remove amplitude projection + rotation; use raw 5D features as embedding"

Architecture:
    features [B, N+1, 5]
        → Linear(5→4) + ReLU  (plain MLP, no L2 norm, no rotation)
        → embedding [B, N+1, 4]

This is intentionally matched to QAPEncoder's OUTPUT DIMENSION (4D) so that
the decoder, critic, and PPO loop are completely unchanged.  The only
difference is the encoder: no unit-norm constraint, no rotation matrix,
no quantum-inspired interpretation.

Parameter count:
    Linear(5→4):  5×4 + 4 = 24 params
    (vs QAPEncoder: 24 proj + 230 rotation = 254 params)

This means the baseline actually has FEWER parameters — which is conservative
and strengthens any result showing QAP-DRL is better.

Usage in train_ablation_n20.py:
    policy = QAPPolicy(encoder_type="baseline")
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from .feature_constructor import FeatureBuilder
from utils.knn import compute_knn


class BaselineEncoder(nn.Module):
    """
    Plain MLP encoder — no amplitude projection, no rotation.

    Maps 5D node features to 4D embeddings using a standard
    linear layer + ReLU activation.  No unit-norm constraint.
    No rotation matrix.  Pure DRL baseline.

    Args:
        input_dim:  feature dimension (default 5, must match QAPEncoder)
        output_dim: embedding dimension (default 4, must match QAPEncoder)

    Parameter count: 5×4 + 4 = 24
    """

    def __init__(self, input_dim: int = 5, output_dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)  # 24 params

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N+1, 5]

        Returns:
            embedding: [B, N+1, D]  — NOT unit-norm, no rotation applied
        """
        return F.relu(self.proj(features))             # [B, N+1, D]


class FullBaselineEncoder(nn.Module):
    """
    Drop-in replacement for FullEncoder that uses BaselineEncoder.

    Identical interface — returns (embedding, features, knn_indices).
    Used by QAPPolicy when encoder_type="baseline".
    """

    def __init__(self, input_dim: int = 5, output_dim: int = 4,
                 knn_k: int = 5):
        super().__init__()
        self.feature_builder   = FeatureBuilder()
        self.baseline_encoder  = BaselineEncoder(input_dim, output_dim)
        self.knn_k             = knn_k

    def forward(self, state: dict):
        """
        Args:
            state: dict with coords [B,N+1,2], demands [B,N+1], capacity

        Returns:
            embedding:   [B, N+1, D]  plain MLP embedding (no norm, no rotation)
            features:    [B, N+1, 5]  raw 5D features
            knn_indices: [B, N+1, k]  spatial kNN (identical to QAP version)
        """
        features    = self.feature_builder(state)                      # [B, N+1, 5]
        embedding   = self.baseline_encoder(features)                  # [B, N+1, D]
        knn_indices = compute_knn(state["coords"], self.knn_k)         # [B, N+1, k]
        return embedding, features, knn_indices
