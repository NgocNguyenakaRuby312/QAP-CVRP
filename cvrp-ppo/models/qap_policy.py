"""
models/qap_policy.py
=====================
QAPPolicy — actor + critic (shared encoder) for PPO.

Parameter budget after Changes 1 + 2 + 3 (§3.3.1 / §3.3.3 / §3.3.4):
    W (amplitude proj): 2×6 = 12  (+2 Change 3, was 10)
    b (proj bias):      2
    MLP rotation:       6×16+16+16×1+1 = 129  (+16 Change 3, was 113)
    W_q:                2×6 = 12   (+4 Change 2, was 8)
    λ:                  1
    μ:                  1          (+1 Change 1)
    Critic MLP:         2→64→1 = 257
    Actor total:        ~157
    Full total:         ~414

Changes vs original:
    Change 1 (§3.3.4 Scoring):
        Score(j) = q·ψ'ⱼ + λ·E_kNN(j) − μ·dist(vₜ, vⱼ)
        μ = learnable nn.Parameter, init 0.5

    Change 2 (§3.3.3 Context):
        ctx ∈ ℝ⁶ = [ψ'_curr(2), cap/C(1), t/N(1), x_curr(1), y_curr(1)]
        Wq ∈ ℝ^{2×6} (was 2×4)
        context_query.forward() returns (query, current_coords)

    Change 3 (§3.3.1 Dynamic feature):
        Feature vector xᵢ(t) ∈ ℝ⁶: adds dist(i, vₜ) as 6th element
        AmplitudeProjection: input_dim 5→6 (+2 params)
        RotationMLP: input_dim 5→6, first layer 5×16→6×16 (+16 params)
        FullEncoder.forward() accepts current_node_coords [B,2] optional arg
        QAPDecoder.rollout() receives encoder ref for per-step re-encoding
        evaluate_actions() re-builds features per step with cur_coords_3d

    evaluate_actions() updated (Changes 1+2+3):
        - features_3d [mb, T, N+1, 6] built per-step with cur_coords_3d
        - psi_prime_3d [mb, T, N+1, 2] encoded per-step
        - ctx [mb, T, 6] with x_curr, y_curr (Change 2)
        - dist_to_nodes [mb, T, N+1] for distance penalty (Change 1)
        - 3-term scoring: context + interference + distance penalty
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
        feature_dim:   node feature size (default 6 — Change 3: was 5)
        hidden_dim:    rotation MLP hidden size (default 16)
        knn_k:         kNN neighbourhood size (default 5)
        lambda_init:   initial hybrid λ (default 0.1)
        mu_init:       initial distance penalty μ (default 0.5)  Change 1
        encoder_type:  "qap" or "baseline"
    """

    def __init__(
        self,
        feature_dim:   int   = 6,          # Change 3: was 5
        hidden_dim:    int   = 16,
        knn_k:         int   = 5,
        lambda_init:   float = 0.1,
        mu_init:       float = 0.5,        # Change 1
        encoder_type:  str   = "qap",
        **kwargs,
    ):
        super().__init__()
        if encoder_type == "baseline":
            self.encoder = FullBaselineEncoder(feature_dim, 2, knn_k)
        else:
            self.encoder = FullEncoder(feature_dim, 2, hidden_dim, knn_k)
        self.encoder_type = encoder_type
        # Change 1+2: context_dim=6, mu_init passed through
        self.decoder = QAPDecoder(context_dim=6, embed_dim=2,
                                  lambda_init=lambda_init, mu_init=mu_init)
        self.critic_head = nn.Sequential(                              # §5: 2→64→1
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def get_value(self, psi_prime: torch.Tensor) -> torch.Tensor:
        """Critic: mean-pool → MLP → V(s).  [B,N+1,2] → [B]"""
        pooled = psi_prime.mean(dim=1)                                 # [B, 2]
        return self.critic_head(pooled).squeeze(-1)                    # [B]

    def forward(self, state: dict, env, deterministic: bool = False):
        """
        Roll out one complete episode.

        Change 3: encoder reference passed to decoder.rollout() so that
        psi_prime is re-computed at each step with the current vehicle
        coordinates as feature[5].

        Returns:
            actions:      [B, T]
            log_probs:    [B, T]
            sum_log_prob: [B]
        """
        # Initial encode — current_node_coords=None → feature[5]=dist(i,depot)
        psi_prime, _, knn_indices = self.encoder(state)                # [B,N+1,2], _, [B,N+1,k]

        # Change 3: pass encoder so rollout() re-encodes per step
        enc_ref = self.encoder if self.encoder_type == "qap" else None
        actions, log_probs, _ = self.decoder.rollout(
            psi_prime, state, knn_indices, env,
            greedy=deterministic,
            encoder=enc_ref,                                           # Change 3
        )
        return actions, log_probs, log_probs.sum(dim=1)

    def evaluate_actions(
        self,
        instance:    dict,            # {coords, demands, capacity}
        actions:     torch.Tensor,    # [B, T]  stored actions
        psi_prime:   torch.Tensor,    # [B, N+1, 2]  initial encode (Change 3: rebuilt per step)
        knn_indices: torch.Tensor,    # [B, N+1, k]  precomputed on spatial coords
    ):
        """
        Re-evaluate stored actions under current policy.

        Vectorized re-implementation across all T steps simultaneously.

        Changes vs original:
            Change 3: features rebuilt per-step with cur_coords_3d [mb,T,2]
                      so that feature[5]=dist(i,vₜ) varies across steps.
                      psi_prime_3d [mb,T,N+1,2] is computed for each step.
            Change 2: ctx [mb,T,6] includes x_curr,y_curr
            Change 1: dist_to_nodes [mb,T,N+1] for μ·dist penalty

        Returns:
            lp_new:  [B, T]   log-prob of each taken action
            entropy: [B]      mean per-step entropy
        """
        mb, T        = actions.shape
        n_customers  = psi_prime.shape[1] - 1
        device       = psi_prime.device

        demands     = instance["demands"]    # [mb, N+1]
        capacity    = instance["capacity"]
        all_coords  = instance["coords"]     # [mb, N+1, 2]

        # ── Precompute all T intermediate states from stored actions ──
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
                vis[:, 0] = False                                       # P7
                cur      = act_t

        # ── Infeasibility mask [mb, T, N+1] ──────────────────────────
        if isinstance(capacity, (int, float)):
            cap_float = float(capacity)
            remaining = cap_float - all_used                           # [mb, T]
            cap_norm  = remaining / max(cap_float, 1e-8)               # [mb, T]
        else:
            cap_t     = capacity.unsqueeze(-1)                         # [mb, 1]
            remaining = cap_t - all_used                               # [mb, T]
            cap_norm  = remaining / cap_t.clamp(min=1e-8)             # [mb, T]

        exceeds  = demands.unsqueeze(1) > remaining.unsqueeze(-1)      # [mb, T, N+1]
        mask_3d  = all_vis | exceeds                                   # [mb, T, N+1]
        mask_3d[:, :, 0] = False                                       # P3
        at_depot_3d = (all_cur == 0)                                   # [mb, T]
        has_cust_3d = (~mask_3d[:, :, 1:]).any(dim=-1)                 # [mb, T]
        mask_3d[:, :, 0] = mask_3d[:, :, 0] | (at_depot_3d & has_cust_3d)

        # ── Current vehicle coordinates for all (instance, step) pairs ─
        # all_coords: [mb, N+1, 2]; all_cur: [mb, T]
        cur_coords_3d = all_coords.gather(
            1, all_cur.unsqueeze(-1).expand(mb, T, 2)                  # [mb, T, 2]
        )                                                               # [mb, T, 2]

        # ── Change 3: Re-build features per step with cur_coords_3d ──
        # For QAP encoder only — baseline encoder does not use Change 3
        if self.encoder_type == "qap":
            # Build features for every (instance, step) pair
            # cur_coords_3d: [mb, T, 2] → loop T steps, build [mb, N+1, 6] each
            # Stack into [mb, T, N+1, 6] then encode → [mb, T, N+1, 2]
            psi_prime_list = []
            for t_idx in range(T):
                cur_c_t  = cur_coords_3d[:, t_idx, :]                  # [mb, 2]
                feats_t  = self.encoder.build_features(instance, cur_c_t)  # [mb, N+1, 6]
                psi_t    = self.encoder.qap_encoder(feats_t)            # [mb, N+1, 2]
                psi_prime_list.append(psi_t)
            # [mb, N+1, 2] × T → stack on dim1 → [mb, T, N+1, 2]
            psi_prime_3d = torch.stack(psi_prime_list, dim=1)          # [mb, T, N+1, 2]
        else:
            # Baseline: use initial psi_prime, broadcast across T
            psi_prime_3d = psi_prime.unsqueeze(1).expand(mb, T, -1, -1)  # [mb,T,N+1,2]

        # ── ψ'_curr for all (instance, step) pairs ────────────────────
        # Gather from per-step psi_prime_3d using all_cur
        cur_exp  = all_cur.unsqueeze(-1).unsqueeze(-1).expand(mb, T, 1, 2)  # [mb,T,1,2]
        psi_curr = psi_prime_3d.gather(2, cur_exp).squeeze(2)           # [mb, T, 2]
        psi_curr = psi_curr.masked_fill(at_depot_3d.unsqueeze(-1), 0.0)

        # ── Change 2: Context [mb, T, 6] → query [mb, T, 2] ──────────
        t_norm = (torch.arange(T, device=device).float() / max(n_customers, 1)
                  ).unsqueeze(0).expand(mb, T)                         # [mb, T]
        ctx = torch.cat([                                              # [mb, T, 6]
            psi_curr,                                                  # [mb, T, 2]
            cap_norm.unsqueeze(-1),                                    # [mb, T, 1]
            t_norm.unsqueeze(-1),                                      # [mb, T, 1]
            cur_coords_3d,                                             # [mb, T, 2]  Change 2
        ], dim=-1)
        query = self.decoder.context_query.Wq(ctx)                     # [mb, T, 2]

        # ── Term 1: Context scores q[b,t] · ψ'[b,t,j] → [mb, T, N+1]
        # Change 3: psi_prime_3d is per-step, not broadcast
        context_scores = torch.einsum('btd,btnd->btn', query, psi_prime_3d)  # [mb,T,N+1]

        # ── Term 2: Interference — computed per step from psi_prime_3d
        # For each step t, E_kNN uses that step's psi_prime_3d[:,t,:,:]
        # Stack per-step interference into [mb, T, N+1]
        interf_list = []
        for t_idx in range(T):
            psi_t   = psi_prime_3d[:, t_idx, :, :]                    # [mb, N+1, 2]
            inf_t   = self.decoder.hybrid._eknn(psi_t, knn_indices)   # [mb, N+1]
            interf_list.append(inf_t)
        interference = torch.stack(interf_list, dim=1)                 # [mb, T, N+1]

        # ── Term 3: Distance penalty [mb, T, N+1]  (Change 1) ─────────
        dist_to_nodes = torch.norm(
            all_coords.unsqueeze(1) - cur_coords_3d.unsqueeze(2),      # [mb, T, N+1, 2]
            p=2, dim=-1                                                 # [mb, T, N+1]
        )

        # ── Full 3-term score ─────────────────────────────────────────
        scores = (
            context_scores                                              # [mb, T, N+1]
            + self.decoder.hybrid.lambda_param * interference          # λ · E_kNN
            - self.decoder.hybrid.mu_param     * dist_to_nodes        # − μ · dist  Change 1
        )
        scores     = scores.masked_fill(mask_3d, -1e9)                # P2
        log_probs_3d = F.log_softmax(scores, dim=-1)                  # [mb, T, N+1]

        # ── Gather log-probs for taken actions ────────────────────────
        lp_new = log_probs_3d.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [mb, T]

        # ── Entropy ───────────────────────────────────────────────────
        ent_flat = Categorical(
            logits=log_probs_3d.reshape(mb * T, n_customers + 1)
        ).entropy()                                                    # [mb*T]
        entropy  = ent_flat.reshape(mb, T).mean(dim=1)                 # [mb]

        return lp_new, entropy


# Backward-compat aliases
CVRPPolicy = QAPPolicy


class CVRPCritic(nn.Module):
    """Standalone value head. ~257 params. No encoder (shared with actor)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, psi_prime: torch.Tensor) -> torch.Tensor:
        """[B, N+1, 2] -> [B]"""
        pooled = psi_prime.mean(dim=1)                                 # [B, 2]
        return self.value_head(pooled).squeeze(-1)                     # [B]
