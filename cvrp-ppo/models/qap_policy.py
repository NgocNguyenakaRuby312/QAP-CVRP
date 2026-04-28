"""
models/qap_policy.py
=====================
QAPPolicy — actor + critic (shared encoder) for PPO.

Phase 2 (4D amplitudes):
    W (amplitude proj): 4×5 + 4 = 24
    MLP rotation:       5×32+32+32×6+6 = 230
    W_q:                4×8 = 32        (context_dim = 4+1+1+2 = 8)
    λ, μ:               2
    Critic MLP:         4→64→1 = 321
    Actor total:        ~288
    Full total:         ~609
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from encoder import FullEncoder
from encoder.baseline_encoder import FullBaselineEncoder
from decoder import QAPDecoder


class QAPPolicy(nn.Module):
    """
    Full actor-critic with shared encoder.

    Args:
        feature_dim:   node feature size (default 5)
        amp_dim:       amplitude dimension (default 4 for Phase 2)
        hidden_dim:    rotation MLP hidden size (default 16)
        knn_k:         kNN neighbourhood size (default 5)
        lambda_init:   initial hybrid λ (default 0.1)
        mu_init:       initial distance penalty μ (default 0.5)
        encoder_type:  "qap" or "baseline"
    """

    def __init__(
        self,
        feature_dim:   int   = 5,
        amp_dim:       int   = 4,         # Phase 2: was 2
        hidden_dim:    int   = 32,
        knn_k:         int   = 5,
        lambda_init:   float = 0.1,
        mu_init:       float = 0.5,
        encoder_type:  str   = "qap",
        **kwargs,
    ):
        super().__init__()
        self.amp_dim = amp_dim
        context_dim = amp_dim + 4                                      # D + cap + t + x + y

        if encoder_type == "baseline":
            self.encoder = FullBaselineEncoder(feature_dim, amp_dim, knn_k)
        else:
            self.encoder = FullEncoder(feature_dim, amp_dim, hidden_dim, knn_k)
        self.encoder_type = encoder_type

        self.decoder = QAPDecoder(context_dim=context_dim, embed_dim=amp_dim,
                                  lambda_init=lambda_init, mu_init=mu_init)
        self.critic_head = nn.Sequential(
            nn.Linear(amp_dim, 64), nn.ReLU(), nn.Linear(64, 1),      # D→64→1
        )

    def get_value(self, psi_prime: torch.Tensor) -> torch.Tensor:
        """Critic: mean-pool → MLP → V(s).  [B,N+1,D] → [B]"""
        pooled = psi_prime.mean(dim=1)                                 # [B, D]
        return self.critic_head(pooled).squeeze(-1)                    # [B]

    def forward(self, state: dict, env, deterministic: bool = False):
        """
        Roll out one complete episode.
        Encoder called ONCE. psi_prime fixed for all decode steps.
        """
        psi_prime, _, knn_indices = self.encoder(state)                # [B,N+1,D]

        actions, log_probs, _ = self.decoder.rollout(
            psi_prime, state, knn_indices, env,
            greedy=deterministic,
        )
        return actions, log_probs, log_probs.sum(dim=1)

    def evaluate_actions(
        self,
        instance:    dict,
        actions:     torch.Tensor,    # [B, T]
        psi_prime:   torch.Tensor,    # [B, N+1, D]
        knn_indices: torch.Tensor,    # [B, N+1, k]
    ):
        """
        Re-evaluate stored actions under current policy.
        Vectorized across all T steps. Dimension-agnostic (D=2 or D=4).
        """
        mb, T        = actions.shape
        D            = psi_prime.shape[-1]                             # 2 or 4
        n_customers  = psi_prime.shape[1] - 1
        device       = psi_prime.device

        demands     = instance["demands"]
        capacity    = instance["capacity"]
        all_coords  = instance["coords"]                               # [mb, N+1, 2]

        # ── Precompute all T intermediate states ─────────────────────
        all_cur  = torch.zeros(mb, T, dtype=torch.long, device=device)
        all_used = torch.zeros(mb, T, device=device)
        all_vis  = torch.zeros(mb, T, n_customers + 1, dtype=torch.bool, device=device)

        cur  = torch.zeros(mb, dtype=torch.long, device=device)
        used = torch.zeros(mb, device=device)
        vis  = torch.zeros(mb, n_customers + 1, dtype=torch.bool, device=device)

        for t in range(T):
            all_cur[:, t]  = cur
            all_used[:, t] = used
            all_vis[:, t]  = vis
            if t < T - 1:
                act_t    = actions[:, t]
                at_depot = (act_t == 0)
                dem_t    = demands.gather(1, act_t.unsqueeze(1)).squeeze(1)
                used     = torch.where(at_depot, torch.zeros_like(used), used + dem_t)
                vis.scatter_(1, act_t.unsqueeze(1), True)
                vis[:, 0] = False
                cur      = act_t

        # ── Infeasibility mask [mb, T, N+1] ──────────────────────────
        if isinstance(capacity, (int, float)):
            cap_float = float(capacity)
            remaining = cap_float - all_used
            cap_norm  = remaining / max(cap_float, 1e-8)
        else:
            cap_t     = capacity.unsqueeze(-1)
            remaining = cap_t - all_used
            cap_norm  = remaining / cap_t.clamp(min=1e-8)

        exceeds  = demands.unsqueeze(1) > remaining.unsqueeze(-1)
        mask_3d  = all_vis | exceeds
        mask_3d[:, :, 0] = False
        at_depot_3d = (all_cur == 0)
        has_cust_3d = (~mask_3d[:, :, 1:]).any(dim=-1)
        mask_3d[:, :, 0] = mask_3d[:, :, 0] | (at_depot_3d & has_cust_3d)

        # ── Current vehicle coordinates ──────────────────────────────
        cur_coords_3d = all_coords.gather(
            1, all_cur.unsqueeze(-1).expand(mb, T, 2)                  # [mb, T, 2]
        )

        # ── psi_prime STATIC — broadcast across T steps ──────────────
        psi_prime_3d = psi_prime.unsqueeze(1).expand(mb, T, -1, -1)    # [mb,T,N+1,D]

        # ── ψ'_curr for all (instance, step) pairs ──────────────────
        cur_exp  = all_cur.unsqueeze(-1).unsqueeze(-1).expand(mb, T, 1, D)  # [mb,T,1,D]
        psi_curr = psi_prime_3d.gather(2, cur_exp).squeeze(2)          # [mb, T, D]
        psi_curr = psi_curr.masked_fill(at_depot_3d.unsqueeze(-1), 0.0)

        # ── Context [mb, T, D+4] → query [mb, T, D] ─────────────────
        t_norm = (torch.arange(T, device=device).float() / max(n_customers, 1)
                  ).unsqueeze(0).expand(mb, T)
        ctx = torch.cat([
            psi_curr,                                                  # [mb, T, D]
            cap_norm.unsqueeze(-1),                                    # [mb, T, 1]
            t_norm.unsqueeze(-1),                                      # [mb, T, 1]
            cur_coords_3d,                                             # [mb, T, 2]
        ], dim=-1)                                                     # [mb, T, D+4]
        query = self.decoder.context_query.Wq(ctx)                     # [mb, T, D]

        # ── Term 1: Context scores ───────────────────────────────────
        context_scores = torch.einsum('btd,btnd->btn', query, psi_prime_3d)  # [mb,T,N+1]

        # ── Term 2: Interference ─────────────────────────────────────
        interf_list = []
        for t_idx in range(T):
            psi_t   = psi_prime_3d[:, t_idx, :, :]                    # [mb, N+1, D]
            inf_t   = self.decoder.hybrid._eknn(psi_t, knn_indices)   # [mb, N+1]
            interf_list.append(inf_t)
        interference = torch.stack(interf_list, dim=1)                 # [mb, T, N+1]

        # ── Term 3: Distance penalty ─────────────────────────────────
        dist_to_nodes = torch.norm(
            all_coords.unsqueeze(1) - cur_coords_3d.unsqueeze(2),
            p=2, dim=-1                                                 # [mb, T, N+1]
        )

        # ── Clamp learnable scalars (must match hybrid_scoring.py) ───
        mu_eff  = torch.clamp(self.decoder.hybrid.mu_param,     min=0.0, max=20.0)
        lam_eff = torch.clamp(self.decoder.hybrid.lambda_param, min=-2.0, max=3.0)
        nu_eff  = torch.clamp(self.decoder.hybrid.nu_param,     min=-2.0, max=3.0)

        # ── Full 4-term score ────────────────────────────────────────
        scores = (
            context_scores
            + lam_eff * interference
            - mu_eff  * dist_to_nodes
        )

        # ── Term 4: demand awareness (nu term) ───────────────────────
        if isinstance(capacity, (int, float)):
            cap_val = max(float(capacity), 1e-8)
        elif hasattr(capacity, 'dim') and capacity.dim() == 0:
            cap_val = max(capacity.item(), 1e-8)
        else:
            cap_val = capacity.clamp(min=1e-8)
            if cap_val.dim() == 1:  # [B] -> scalar (use first, all same)
                cap_val = cap_val[0].item()
        demand_ratio = demands.unsqueeze(1).expand(mb, T, -1) / cap_val  # [mb, T, N+1]
        scores = scores + nu_eff * demand_ratio
        scores       = scores.masked_fill(mask_3d, -1e9)
        log_probs_3d = F.log_softmax(scores, dim=-1)                  # [mb, T, N+1]

        # ── Gather log-probs for taken actions ───────────────────────
        lp_new = log_probs_3d.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        # ── Entropy ──────────────────────────────────────────────────
        ent_flat = Categorical(
            logits=log_probs_3d.reshape(mb * T, n_customers + 1)
        ).entropy()
        entropy  = ent_flat.reshape(mb, T).mean(dim=1)                 # [mb]

        return lp_new, entropy


CVRPPolicy = QAPPolicy


class CVRPCritic(nn.Module):
    """Standalone value head."""

    def __init__(self, amp_dim: int = 4, **kwargs):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(amp_dim, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, psi_prime: torch.Tensor) -> torch.Tensor:
        pooled = psi_prime.mean(dim=1)                                 # [B, D]
        return self.value_head(pooled).squeeze(-1)                     # [B]
