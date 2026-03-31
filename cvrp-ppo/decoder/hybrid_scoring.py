"""
decoder/hybrid_scoring.py
==========================
Step 6 — Hybrid Scoring → Select Node.

    Score(j) = q · ψ'ⱼ  +  λ · Σ_{i∈kNN(j)} (ψ'ᵢ · ψ'ⱼ)           # Eq §3.X.7
    P(j)     = softmax( Score(j) )     [infeasible → −1e9 BEFORE softmax]

λ is a learnable nn.Parameter, initialised to 0.1.                     # §15: self.lambda_param
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridScoring(nn.Module):
    """
    Hybrid scoring: context attention + kNN interference.

    Args:
        lambda_init: initial value of learnable λ (default 0.1)
    """

    def __init__(self, lambda_init: float = 0.1):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))    # P12: nn.Parameter

    # ── EkNN interference ────────────────────────────────────────────

    def _eknn(self, psi_prime: torch.Tensor, knn_indices: torch.Tensor) -> torch.Tensor:
        """
        EkNN(j) = Σ_{i∈kNN(j)} ψ'ᵢ · ψ'ⱼ                           # Eq: E_kNN(j)

        Args:
            psi_prime:   [B, N+1, 2]
            knn_indices: [B, N+1, k]  precomputed on spatial coords

        Returns:
            interference: [B, N+1]
        """
        B, Np1, k = knn_indices.shape

        # Gather neighbour embeddings
        idx = knn_indices.unsqueeze(-1).expand(B, Np1, k, 2)          # [B, N+1, k, 2]
        neighbors = psi_prime.unsqueeze(1).expand(B, Np1, Np1, 2) \
                    .gather(2, idx)                                    # [B, N+1, k, 2]

        # Dot product of each node j with each of its k neighbours
        psi_exp = psi_prime.unsqueeze(2).expand(B, Np1, k, 2)         # [B, N+1, k, 2]
        dots = (psi_exp * neighbors).sum(dim=-1)                       # [B, N+1, k]
        return dots.sum(dim=-1)                                        # [B, N+1]

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        query:       torch.Tensor,    # [B, 2]
        psi_prime:   torch.Tensor,    # [B, N+1, 2]
        knn_indices: torch.Tensor,    # [B, N+1, k]  precomputed on spatial coords
        mask:        torch.Tensor,    # [B, N+1]  True = INFEASIBLE
    ) -> torch.Tensor:
        """
        Returns:
            log_probs: [B, N+1]  log-softmax over valid nodes
        """
        # Context scores: q · ψ'ⱼ                                     # Eq: S_context(j)
        context_scores = (psi_prime * query.unsqueeze(1)).sum(dim=-1)  # [B, N+1]

        # Interference: Σ_{i∈kNN(j)} ψ'ᵢ · ψ'ⱼ                      # Eq: E_kNN(j)
        interference = self._eknn(psi_prime, knn_indices)              # [B, N+1]

        # Hybrid score                                                 # Eq: Score(j) = S + λ·E
        scores = context_scores + self.lambda_param * interference     # [B, N+1]

        # Mask infeasible nodes BEFORE log_softmax                     # P2: -1e9 BEFORE softmax
        scores = scores.masked_fill(mask, -1e9)                        # [B, N+1]

        log_probs = F.log_softmax(scores, dim=-1)                     # [B, N+1]
        return log_probs
