"""
decoder/context_query.py
=========================
Step 5 — Context Vector → Query.

Assembles the decoder context from the current partial solution state and
projects it into a 2-D query vector in the same space as the node embeddings:

    ctx   = [ψ'_curr(0), ψ'_curr(1),  cap/C,  t/N,  x_curr,  y_curr]  ∈ ℝ⁶  # Change 2
    query = Wq · ctx                                                     ∈ ℝ²

ψ'_curr = **zero vector** when current position is depot (index 0).

Change 2 (spatial grounding, §3.3.3):
    Context vector extended from ℝ⁴ → ℝ⁶ by appending current vehicle
    Euclidean coordinates (x_curr, y_curr) ∈ [0,1]².

    Motivation: the original ℝ⁴ context encodes vehicle state as (amplitude of
    current node, remaining capacity, progress fraction). At the depot ψ'_curr=[0,0],
    so the query vector is determined entirely by the capacity column of Wq, which
    points in a direction unrelated to node proximity. Adding raw coordinates gives
    Wq direct access to "where am I in physical space?", allowing it to learn query
    directions that favour nodes near the current position.

    At the depot x_curr = depot_x, y_curr = depot_y (from state["coords"][:,0,:]).
    Wq expands from ℝ^{2×4} (8 params) to ℝ^{2×6} (12 params, +4 params total).

Also returns current_coords [B, 2] for use in HybridScoring (Change 1 distance term).
"""

import torch
import torch.nn as nn


class ContextAndQuery(nn.Module):
    """
    Builds ctx and projects to query via Wq ∈ ℝ^{2×6}, NO bias.

    Args:
        context_dim: context vector size (default 6 — includes x_curr, y_curr)
        embed_dim:   query / amplitude space size (default 2)
    """

    def __init__(self, context_dim: int = 6, embed_dim: int = 2):
        super().__init__()
        # Change 2: Wq is now 2×6 (was 2×4). The extra 2 columns correspond to
        # x_curr and y_curr. All 12 entries are jointly optimised by PPO gradient.
        self.Wq = nn.Linear(context_dim, embed_dim, bias=False)        # W_q ∈ ℝ^{2×6}
        nn.init.xavier_uniform_(self.Wq.weight)

    def forward(
        self,
        state:       dict,
        psi_prime:   torch.Tensor,    # [B, N+1, 2]
        step:        int,
        n_customers: int,
    ) -> tuple:
        """
        Args:
            state:       current environment state dict
            psi_prime:   [B, N+1, 2]  encoder output
            step:        current step t (0-indexed)
            n_customers: total customers N

        Returns:
            query:          [B, 2]
            current_coords: [B, 2]  Change 1: passed to HybridScoring for dist penalty
        """
        B      = psi_prime.shape[0]
        device = psi_prime.device

        # ψ'_curr: embedding of last visited node                     # Eq: ψ'_curr
        cur_idx  = state["current_node"]                               # [B]
        psi_curr = psi_prime[torch.arange(B, device=device), cur_idx]  # [B, 2]

        # Zero vector when at depot (index 0)                          # §4: ψ'_curr = 0 at depot
        at_depot = (cur_idx == 0).unsqueeze(-1)                        # [B, 1]
        psi_curr = psi_curr.masked_fill(at_depot, 0.0)                 # [B, 2]

        # Remaining capacity / C                                       # Eq: cap_t / C
        capacity = state["capacity"]
        used     = state["used_capacity"]                              # [B]
        if isinstance(capacity, (int, float)):
            cap_norm = (capacity - used) / max(capacity, 1e-8)         # [B]
        elif capacity.dim() == 0:
            cap_norm = (capacity.item() - used) / max(capacity.item(), 1e-8)
        else:
            cap_norm = (capacity - used) / capacity.clamp(min=1e-8)    # [B]

        # Normalised step  t / N                                       # Eq: t / N
        t_norm = torch.full((B,), step / max(n_customers, 1), device=device)

        # ── Change 2: Current vehicle Euclidean coordinates ──────────
        # all_coords: [B, N+1, 2]; current node index: [B]
        # current_coords: [B, 2] — raw (x,y) in [0,1]²
        all_coords = state["coords"]                                   # [B, N+1, 2]
        current_coords = all_coords[
            torch.arange(B, device=device), cur_idx                    # [B, 2]
        ]

        # ctx ∈ ℝ⁶  = [ψ'_curr(2), cap/C(1), t/N(1), x_curr(1), y_curr(1)]  # [B, 6]
        ctx = torch.cat([
            psi_curr,                                                  # [B, 2]
            cap_norm.unsqueeze(1),                                     # [B, 1]
            t_norm.unsqueeze(1),                                       # [B, 1]
            current_coords,                                            # [B, 2]  Change 2
        ], dim=1)                                                      # [B, 6]

        query = self.Wq(ctx)                                           # [B, 2]
        return query, current_coords                                   # Change 1+2: return coords
