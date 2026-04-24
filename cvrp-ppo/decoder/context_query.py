"""
decoder/context_query.py
=========================
Step 5 — Context Vector → Query.

Phase 2 (4D amplitudes):
    ctx   = [ψ'_curr(4), cap/C(1), t/N(1), x_curr(1), y_curr(1)]  ∈ ℝ⁸
    query = Wq · ctx                                                 ∈ ℝ⁴
    Wq ∈ ℝ^{4×8} = 32 params

Phase 1 (2D):
    ctx = [ψ'_curr(2), cap/C(1), t/N(1), x_curr(1), y_curr(1)]  ∈ ℝ⁶
    Wq ∈ ℝ^{2×6} = 12 params

ψ'_curr = **zero vector** when current position is depot (index 0).
Also returns current_coords [B, 2] for HybridScoring distance penalty.
"""

import torch
import torch.nn as nn


class ContextAndQuery(nn.Module):
    """
    Builds ctx and projects to query via Wq, NO bias.

    Args:
        context_dim: context vector size (default 8 for Phase 2: 4+1+1+2)
        embed_dim:   query / amplitude space size (default 4 for Phase 2)
    """

    def __init__(self, context_dim: int = 8, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(context_dim, embed_dim, bias=False)        # W_q ∈ ℝ^{D×ctx}
        nn.init.xavier_uniform_(self.Wq.weight)

    def forward(
        self,
        state:       dict,
        psi_prime:   torch.Tensor,    # [B, N+1, D]
        step:        int,
        n_customers: int,
    ) -> tuple:
        """
        Returns:
            query:          [B, D]
            current_coords: [B, 2]
        """
        B      = psi_prime.shape[0]
        D      = psi_prime.shape[-1]                                   # 4 or 2
        device = psi_prime.device

        cur_idx  = state["current_node"]                               # [B]
        psi_curr = psi_prime[torch.arange(B, device=device), cur_idx]  # [B, D]

        at_depot = (cur_idx == 0).unsqueeze(-1)                        # [B, 1]
        psi_curr = psi_curr.masked_fill(at_depot, 0.0)                 # [B, D]

        capacity = state["capacity"]
        used     = state["used_capacity"]                              # [B]
        if isinstance(capacity, (int, float)):
            cap_norm = (capacity - used) / max(capacity, 1e-8)
        elif capacity.dim() == 0:
            cap_norm = (capacity.item() - used) / max(capacity.item(), 1e-8)
        else:
            cap_norm = (capacity - used) / capacity.clamp(min=1e-8)

        t_norm = torch.full((B,), step / max(n_customers, 1), device=device)

        all_coords = state["coords"]                                   # [B, N+1, 2]
        current_coords = all_coords[
            torch.arange(B, device=device), cur_idx                    # [B, 2]
        ]

        # ctx ∈ ℝ^{D+4} = [ψ'_curr(D), cap/C(1), t/N(1), x_curr(1), y_curr(1)]
        ctx = torch.cat([
            psi_curr,                                                  # [B, D]
            cap_norm.unsqueeze(1),                                     # [B, 1]
            t_norm.unsqueeze(1),                                       # [B, 1]
            current_coords,                                            # [B, 2]
        ], dim=1)                                                      # [B, D+4]

        query = self.Wq(ctx)                                           # [B, D]
        return query, current_coords
