"""
decoder/hybrid_scoring.py
==========================
Step 6 — Hybrid Scoring → Select Node.

    Score(j) = q · ψ'ⱼ  +  λ · Σ_{i∈kNN(j)} (ψ'ᵢ · ψ'ⱼ)  −  μ · dist(vₜ, vⱼ)

    P(j)     = softmax( Score(j) )     [infeasible → −1e9 BEFORE softmax]

λ is a learnable nn.Parameter, initialised to 0.1.                     # §15: self.lambda_param
μ is a learnable nn.Parameter, initialised to 0.5.                     # Change 1: distance penalty

Change 1 (proximity penalty, §3.3.4):
    Added third term: −μ · dist(vₜ, vⱼ)
    vₜ   = current vehicle position (depot coords when at depot)
    vⱼ   = candidate node j coordinates
    μ    = learnable scalar ∈ ℝ, init 0.5

    Motivation: the two existing terms (attention + interference) are both purely
    amplitude-space signals — they measure angular alignment on S¹. Neither term has
    any knowledge of physical Euclidean proximity between the current vehicle position
    and candidate nodes. This caused the model to choose distant nodes with favourable
    amplitude geometry over nearer nodes with less-favourable geometry (e.g. choosing
    C1 at dist=0.39 over C5 at dist=0.09 in the 5-node demo). The proximity penalty
    directly deducts travel cost from the logit before softmax, biasing selection toward
    nodes that are both geometrically coherent AND physically nearby.

    forward() now requires an additional argument:
        current_coords: [B, 2]  Euclidean coordinates of the vehicle's current node.
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
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))    # P12: nn.Parameter
        self.mu_param     = nn.Parameter(torch.tensor(mu_init))        # Change 1: μ for dist

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

        # Gather neighbour embeddings via flat index — avoids [B,N+1,N+1,2] expansion
        flat_idx   = knn_indices.reshape(B, -1)                        # [B, (N+1)*k]
        flat_idx_e = flat_idx.unsqueeze(-1).expand(B, Np1 * k, 2)     # [B, (N+1)*k, 2]
        neighbors  = psi_prime.gather(1, flat_idx_e)                   # [B, (N+1)*k, 2]
        neighbors  = neighbors.view(B, Np1, k, 2)                      # [B, N+1, k, 2]

        # Dot product of each node j with each of its k neighbours
        psi_exp = psi_prime.unsqueeze(2).expand(B, Np1, k, 2)         # [B, N+1, k, 2]
        dots = (psi_exp * neighbors).sum(dim=-1)                       # [B, N+1, k]
        return dots.sum(dim=-1)                                        # [B, N+1]

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        query:          torch.Tensor,    # [B, 2]
        psi_prime:      torch.Tensor,    # [B, N+1, 2]
        knn_indices:    torch.Tensor,    # [B, N+1, k]  precomputed on spatial coords
        mask:           torch.Tensor,    # [B, N+1]  True = INFEASIBLE
        current_coords: torch.Tensor,    # [B, 2]   Change 1: current vehicle position
        all_coords:     torch.Tensor,    # [B, N+1, 2]  Change 1: all node coordinates
    ) -> torch.Tensor:
        """
        Returns:
            log_probs: [B, N+1]  log-softmax over valid nodes
        """
        # ── Term 1: Context scores: q · ψ'ⱼ ─────────────────────────  # Eq: S_context(j)
        context_scores = (psi_prime * query.unsqueeze(1)).sum(dim=-1)  # [B, N+1]

        # ── Term 2: Interference: Σ_{i∈kNN(j)} ψ'ᵢ · ψ'ⱼ ────────────  # Eq: E_kNN(j)
        interference = self._eknn(psi_prime, knn_indices)              # [B, N+1]

        # ── Term 3: Distance penalty − μ · dist(vₜ, vⱼ) ─────────────  # Change 1
        # current_coords: [B, 2] → expand to [B, 1, 2] for broadcast
        # all_coords:     [B, N+1, 2]
        # dist:           [B, N+1]  Euclidean distance from current to each node
        dist_to_nodes = torch.norm(
            all_coords - current_coords.unsqueeze(1), p=2, dim=-1      # [B, N+1]
        )

        # ── Hybrid score ──────────────────────────────────────────────  # Eq: Score(j)
        scores = (
            context_scores                                              # q · ψ'ⱼ
            + self.lambda_param * interference                          # λ · E_kNN(j)
            - self.mu_param     * dist_to_nodes                        # − μ · dist(vₜ,vⱼ)
        )                                                               # [B, N+1]

        # ── Mask infeasible nodes BEFORE log_softmax ──────────────────  # P2
        scores = scores.masked_fill(mask, -1e9)                        # [B, N+1]

        log_probs = F.log_softmax(scores, dim=-1)                     # [B, N+1]
        return log_probs
