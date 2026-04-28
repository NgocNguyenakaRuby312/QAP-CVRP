"""
decoder/hybrid_scoring.py
==========================
Step 6 -- Hybrid Scoring -> Select Node.

    Score(j) = q*psi'_j + lam*E_kNN(j) - mu*dist(vt,vj) + nu*(d_j/C)

4 terms:
  1. Context attention:  q . psi'_j
  2. kNN interference:   lam * Sum_kNN(psi'_i . psi'_j)
  3. Distance penalty:   -mu * dist(vt, vj)
  4. Demand awareness:   +nu * (d_j / C)      [Option 3: prefer high-demand nodes]

Dimension-agnostic: works with psi in R^2 (Phase 1) or R^4 (Phase 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridScoring(nn.Module):
    """
    Hybrid scoring: context attention + kNN interference + distance + demand.

    Args:
        lambda_init: initial value of learnable lambda (default 0.1)
        mu_init:     initial value of learnable mu distance penalty (default 0.5)
        nu_init:     initial value of learnable nu demand bonus (default 0.0)
    """

    def __init__(self, lambda_init: float = 0.1, mu_init: float = 0.5,
                 nu_init: float = 0.0):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.mu_param     = nn.Parameter(torch.tensor(mu_init))
        self.nu_param     = nn.Parameter(torch.tensor(nu_init))        # demand-awareness

    def _eknn(self, psi_prime: torch.Tensor, knn_indices: torch.Tensor) -> torch.Tensor:
        """
        EkNN(j) = Sum_{i in kNN(j)} psi'_i . psi'_j

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
        demands:        torch.Tensor = None,  # [B, N+1] optional for nu term
        capacity:       float = 1.0,          # scalar capacity for nu term
    ) -> torch.Tensor:
        """
        Returns:
            log_probs: [B, N+1]  log-softmax over valid nodes
        """
        # Term 1: context attention
        context_scores = (psi_prime * query.unsqueeze(1)).sum(dim=-1)  # [B, N+1]

        # Term 2: kNN interference
        interference = self._eknn(psi_prime, knn_indices)              # [B, N+1]

        # Term 3: distance penalty
        dist_to_nodes = torch.norm(
            all_coords - current_coords.unsqueeze(1), p=2, dim=-1      # [B, N+1]
        )

        # Clamp learnable scalars
        mu_eff  = torch.clamp(self.mu_param,     min=0.0, max=20.0)
        lam_eff = torch.clamp(self.lambda_param, min=-2.0, max=3.0)
        nu_eff  = torch.clamp(self.nu_param,     min=-2.0, max=3.0)

        # Combine 3 main terms
        scores = (
            context_scores
            + lam_eff * interference
            - mu_eff  * dist_to_nodes
        )                                                               # [B, N+1]

        # Term 4: demand awareness (Option 3)
        if demands is not None:
            if isinstance(capacity, (int, float)):
                cap_val = max(float(capacity), 1e-8)
                demand_ratio = demands / cap_val                       # [B, N+1]
            elif hasattr(capacity, 'dim') and capacity.dim() == 0:
                cap_val = max(capacity.item(), 1e-8)
                demand_ratio = demands / cap_val                       # [B, N+1]
            else:
                # capacity is [B] tensor — unsqueeze for broadcasting
                cap_val = capacity.clamp(min=1e-8).unsqueeze(-1)       # [B, 1]
                demand_ratio = demands / cap_val                       # [B, N+1]
            scores = scores + nu_eff * demand_ratio

        # Mask infeasible nodes BEFORE log_softmax
        scores = scores.masked_fill(mask, -1e9)
        log_probs = F.log_softmax(scores, dim=-1)                     # [B, N+1]
        return log_probs
