"""
decoder/hybrid_scoring.py
==========================
Step 6 — Hybrid Scoring → Select Node.

    Score(j) = q · ψ'ⱼ  +  λ · Σ_{i∈kNN(j)} (ψ'ᵢ · ψ'ⱼ)  −  μ · dist(vₜ, vⱼ)

Dimension-agnostic: works with ψ ∈ ℝ² (Phase 1) or ψ ∈ ℝ⁴ (Phase 2).
Dot products and gathers automatically adapt to the amplitude dimension D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridScoring(nn.Module):
    """
    Hybrid scoring: context attention + kNN interference + distance penalty.

    Args:
        lambda_init: initial value of learnable λ (default 0.1)
        mu_init:     initial value of learnable μ distance penalty (default 0.5)
    """

    def __init__(self, lambda_init: float = 0.1, mu_init: float = 0.5):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.mu_param     = nn.Parameter(torch.tensor(mu_init))

    def _eknn(self, psi_prime: torch.Tensor, knn_indices: torch.Tensor) -> torch.Tensor:
        """
        EkNN(j) = Σ_{i∈kNN(j)} ψ'ᵢ · ψ'ⱼ                           # Eq: E_kNN(j)

        Dimension-agnostic: works for any D.

        Args:
            psi_prime:   [B, N+1, D]
            knn_indices: [B, N+1, k]

        Returns:
            interference: [B, N+1]
        """
        B, Np1, k = knn_indices.shape
        D = psi_prime.shape[-1]                                        # 2 or 4

        flat_idx   = knn_indices.reshape(B, -1)                        # [B, (N+1)*k]
        flat_idx_e = flat_idx.unsqueeze(-1).expand(B, Np1 * k, D)     # [B, (N+1)*k, D]
        neighbors  = psi_prime.gather(1, flat_idx_e)                   # [B, (N+1)*k, D]
        neighbors  = neighbors.view(B, Np1, k, D)                      # [B, N+1, k, D]

        psi_exp = psi_prime.unsqueeze(2).expand(B, Np1, k, D)         # [B, N+1, k, D]
        dots = (psi_exp * neighbors).sum(dim=-1)                       # [B, N+1, k]
        return dots.sum(dim=-1)                                        # [B, N+1]

    def forward(
        self,
        query:          torch.Tensor,    # [B, D]
        psi_prime:      torch.Tensor,    # [B, N+1, D]
        knn_indices:    torch.Tensor,    # [B, N+1, k]
        mask:           torch.Tensor,    # [B, N+1]
        current_coords: torch.Tensor,    # [B, 2]
        all_coords:     torch.Tensor,    # [B, N+1, 2]
    ) -> torch.Tensor:
        """
        Returns:
            log_probs: [B, N+1]  log-softmax over valid nodes
        """
        context_scores = (psi_prime * query.unsqueeze(1)).sum(dim=-1)  # [B, N+1]

        interference = self._eknn(psi_prime, knn_indices)              # [B, N+1]

        dist_to_nodes = torch.norm(
            all_coords - current_coords.unsqueeze(1), p=2, dim=-1      # [B, N+1]
        )

        mu_eff  = torch.clamp(self.mu_param,     min=0.0, max=10.0)
        lam_eff = torch.clamp(self.lambda_param, min=-0.5, max=3.0)

        scores = (
            context_scores
            + lam_eff * interference
            - mu_eff  * dist_to_nodes
        )                                                               # [B, N+1]

        scores = scores.masked_fill(mask, -1e9)
        log_probs = F.log_softmax(scores, dim=-1)                     # [B, N+1]
        return log_probs
