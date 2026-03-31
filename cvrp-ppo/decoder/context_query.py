"""
decoder/context_query.py
=========================
Step 5 — Context Vector → Query.

Assembles the decoder context from the current partial solution state and
projects it into a 2-D query vector in the same space as the node embeddings:

    ctx   = [ψ'_curr(0), ψ'_curr(1),  cap/C,  t/N]  ∈ ℝ⁴            # Eq §3.X.6
    query = Wq · ctx                                   ∈ ℝ²

ψ'_curr = **zero vector** when current position is depot (index 0).
"""

import torch
import torch.nn as nn


class ContextAndQuery(nn.Module):
    """
    Builds ctx and projects to query via Wq ∈ ℝ^{2×4}, NO bias.

    Args:
        context_dim: context vector size (default 4)
        embed_dim:   query / amplitude space size (default 2)
    """

    def __init__(self, context_dim: int = 4, embed_dim: int = 2):
        super().__init__()
        self.Wq = nn.Linear(context_dim, embed_dim, bias=False)        # W_q ∈ ℝ^{2×4}
        nn.init.xavier_uniform_(self.Wq.weight)

    def forward(
        self,
        state:       dict,
        psi_prime:   torch.Tensor,    # [B, N+1, 2]
        step:        int,
        n_customers: int,
    ) -> torch.Tensor:
        """
        Args:
            state:       current environment state dict
            psi_prime:   [B, N+1, 2]  encoder output
            step:        current step t (0-indexed)
            n_customers: total customers N

        Returns:
            query: [B, 2]
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

        # ctx ∈ ℝ⁴  = [ψ'_curr(2), cap/C(1), t/N(1)]                 # [B, 4]
        ctx   = torch.cat([psi_curr, cap_norm.unsqueeze(1), t_norm.unsqueeze(1)], dim=1)
        query = self.Wq(ctx)                                           # [B, 2]
        return query
