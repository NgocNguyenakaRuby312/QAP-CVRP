"""
encoder/feature_constructor.py
===============================
Step 2 — Feature Construction.

Builds the 6-dimensional node feature vector for every node (depot + customers):

    xᵢ(t) = [dᵢ/C, ‖i−depot‖, xᵢ, yᵢ, atan2(Δy,Δx)/π, dist(i,vₜ)]  ∈ ℝ⁶  # Eq §3.X.3

Feature [0]: demand normalised by vehicle capacity — depot = 0
Feature [1]: Euclidean distance from node to depot (raw, not normalised)
Feature [2]: raw x-coordinate (already in [0, 1] from data generation)
Feature [3]: raw y-coordinate (already in [0, 1] from data generation)
Feature [4]: polar angle relative to depot, divided by π → ∈ [−1, 1]
Feature [5]: Euclidean distance from node to current vehicle position (Change 3)

Features [0]–[4] are STATIC — computed once and reused.
Feature [5] is DYNAMIC — recomputed at every decoding step as the vehicle moves.

Change 3 (§3.3.1, May 2026):
    Added 6th feature dist(i, vₜ) = ‖(xᵢ,yᵢ) − (x_vₜ, y_vₜ)‖₂.
    When current_node_coords is None (initial encode before decoding loop),
    feature[5] falls back to feature[1] (dist to depot).
    +2 params in AmplitudeProjection (W: 2×5→2×6), +16 in RotationMLP (5×16→6×16).
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class FeatureBuilder(nn.Module):
    """
    Constructs the 6-D feature tensor from instance data.

    Change 3: accepts optional current_node_coords [B, 2] for dynamic
    feature[5] = dist(i, current_vehicle). Falls back to dist(i, depot)
    when current_node_coords is None.

    Requires state dict keys: ``coords``, ``demands``, ``capacity``.
    """

    feature_dim: int = 6                                               # Change 3: was 5

    def __init__(self):
        super().__init__()

    def forward(
        self,
        state: dict,
        current_node_coords: Optional[torch.Tensor] = None,           # [B, 2]  Change 3
    ) -> torch.Tensor:
        """
        Args:
            state: dict with keys
                coords   [B, N+1, 2]  depot at index 0
                demands  [B, N+1]     depot demand = 0
                capacity scalar, 0-d tensor, or [B] tensor
            current_node_coords: [B, 2]  vehicle position for feature[5].
                If None, feature[5] = dist(i, depot) (fallback).

        Returns:
            features: [B, N+1, 6]  order: [d/C, dist_depot, x, y, angle/π, dist_curr]
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
        feat_dist_depot = diff.norm(dim=-1)                            # [B, N+1]

        # ── Feature [2]: xᵢ (raw coordinate) ────────────────────────  # Eq: x_i[2]
        feat_x = coords[:, :, 0]                                      # [B, N+1]

        # ── Feature [3]: yᵢ (raw coordinate) ────────────────────────  # Eq: x_i[3]
        feat_y = coords[:, :, 1]                                      # [B, N+1]

        # ── Feature [4]: atan2(Δy, Δx) / π ──────────────────────────  # Eq: x_i[4]
        feat_angle = torch.atan2(diff[:, :, 1], diff[:, :, 0]) / math.pi  # [B, N+1]

        # ── Feature [5]: dist(i, vₜ) — dynamic proximity ────────────  # Eq: x_i[5]  Change 3
        if current_node_coords is not None:
            # current_node_coords: [B, 2] → [B, 1, 2] for broadcast
            feat_dist_curr = torch.norm(                               # [B, N+1]
                coords - current_node_coords.unsqueeze(1),             # [B, N+1, 2]
                p=2, dim=-1,
            )
        else:
            # Fallback: use dist to depot (= feature[1])
            feat_dist_curr = feat_dist_depot                           # [B, N+1]

        features = torch.stack(
            [feat_demand, feat_dist_depot, feat_x, feat_y,
             feat_angle, feat_dist_curr],                              # Change 3: 6 features
            dim=-1,
        )                                                              # [B, N+1, 6]

        return features
