"""
encoder/feature_constructor.py
===============================
Step 2 — Feature Construction.

Builds the 5-dimensional node feature vector for every node (depot + customers):

    xᵢ = [dᵢ/C,  ‖i−depot‖,  xᵢ,  yᵢ,  atan2(Δy,Δx)/π]  ∈ ℝ⁵     # Eq §3.X.3
          [0]       [1]         [2]   [3]        [4]

Feature [0]: demand normalised by vehicle capacity — depot = 0
Feature [1]: Euclidean distance from node to depot (raw, not normalised)
Feature [2]: raw x-coordinate (already in [0, 1] from data generation)
Feature [3]: raw y-coordinate (already in [0, 1] from data generation)
Feature [4]: polar angle relative to depot, divided by π → ∈ [−1, 1]
"""

import math
import torch
import torch.nn as nn


class FeatureBuilder(nn.Module):
    """
    Stateless module that constructs the 5-D feature tensor from instance data.

    Requires state dict keys: ``coords``, ``demands``, ``capacity``.
    """

    feature_dim: int = 5

    def __init__(self):
        super().__init__()

    def forward(self, state: dict) -> torch.Tensor:
        """
        Args:
            state: dict with keys
                coords   [B, N+1, 2]  depot at index 0
                demands  [B, N+1]     depot demand = 0
                capacity scalar, 0-d tensor, or [B] tensor

        Returns:
            features: [B, N+1, 5]  order: [d/C, dist, x, y, angle/π]
        """
        coords   = state["coords"]                                    # [B, N+1, 2]
        demands  = state["demands"]                                    # [B, N+1]
        capacity = state["capacity"]

        depot_xy = coords[:, 0:1, :]                                   # [B, 1, 2]
        diff     = coords - depot_xy                                   # [B, N+1, 2]

        # ── Feature [0]: dᵢ / C ─────────────────────────────────────  # Eq: x_i[0]
        if isinstance(capacity, (int, float)):
            feat_demand = demands / capacity                           # [B, N+1]
        elif capacity.dim() == 0:
            feat_demand = demands / capacity.item()                    # [B, N+1]
        else:
            feat_demand = demands / capacity.unsqueeze(-1)             # [B, N+1]

        # ── Feature [1]: ‖i − depot‖ (raw distance) ─────────────────  # Eq: x_i[1]
        feat_dist = diff.norm(dim=-1)                                  # [B, N+1]

        # ── Feature [2]: xᵢ (raw coordinate) ────────────────────────  # Eq: x_i[2]
        feat_x = coords[:, :, 0]                                      # [B, N+1]

        # ── Feature [3]: yᵢ (raw coordinate) ────────────────────────  # Eq: x_i[3]
        feat_y = coords[:, :, 1]                                      # [B, N+1]

        # ── Feature [4]: atan2(Δy, Δx) / π ──────────────────────────  # Eq: x_i[4]
        feat_angle = torch.atan2(diff[:, :, 1], diff[:, :, 0]) / math.pi  # [B, N+1] ∈ [-1, 1]

        features = torch.stack(
            [feat_demand, feat_dist, feat_x, feat_y, feat_angle], dim=-1
        )                                                              # [B, N+1, 5]

        return features
