"""
models/qap_policy.py
=====================
QAPPolicy — actor + critic (shared encoder) for PPO.

Parameter budget (§5):
    Actor  ~134:  W(10) + b(2) + MLP(113) + W_q(8) + λ(1)
    Critic ~257:  Linear(2→64) + Linear(64→1)
    Total  ~391
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
        hidden_dim:    rotation MLP hidden size (default 16)
        knn_k:         kNN neighbourhood size (default 5)
        lambda_init:   initial hybrid λ (default 0.1)
    """

    def __init__(
        self,
        feature_dim:   int   = 5,
        hidden_dim:    int   = 16,
        knn_k:         int   = 5,
        lambda_init:   float = 0.1,
        encoder_type:  str   = "qap",   # "qap" = full QAP-DRL  |  "baseline" = plain MLP (ablation b)
        **kwargs,
    ):
        super().__init__()
        if encoder_type == "baseline":
            self.encoder = FullBaselineEncoder(feature_dim, 2, knn_k)
        else:
            self.encoder = FullEncoder(feature_dim, 2, hidden_dim, knn_k)
        self.encoder_type = encoder_type
        self.decoder = QAPDecoder(4, 2, lambda_init)
        self.critic_head = nn.Sequential(                              # §5: 2→64→1
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def get_value(self, psi_prime: torch.Tensor) -> torch.Tensor:
        """Critic: mean-pool → MLP → V(s).  [B,N+1,2] → [B]"""
        pooled = psi_prime.mean(dim=1)                                 # [B, 2]
        return self.critic_head(pooled).squeeze(-1)                    # [B]

    def forward(self, state: dict, env, deterministic: bool = False):
        """
        Roll out one complete episode. Delegates to decoder.rollout (BUG H2).

        Returns:
            actions:      [B, T]
            log_probs:    [B, T]
            sum_log_prob: [B]
        """
        psi_prime, _, knn_indices = self.encoder(state)
        actions, log_probs, _ = self.decoder.rollout(
            psi_prime, state, knn_indices, env, greedy=deterministic
        )
        return actions, log_probs, log_probs.sum(dim=1)

    def evaluate_actions(
        self,
        instance:    dict,            # {coords, demands, capacity} — raw instance dict
        actions:     torch.Tensor,    # [B, T]  stored actions
        psi_prime:   torch.Tensor,    # [B, N+1, 2]  pass in to avoid double encode (BUG H1)
        knn_indices: torch.Tensor,    # [B, N+1, k]
    ):
        """
        Re-evaluate stored actions under current policy.

        Vectorized: precomputes all T intermediate states from stored actions,
        then evaluates all T decode steps in one batched pass [B*T, ...].
        Eliminates T sequential env.step() calls per minibatch update.

        Returns:
            lp_new:  [B, T]   log-prob of each taken action
            entropy: [B]      mean per-step entropy
        """
        mb, T        = actions.shape
        n_customers  = psi_prime.shape[1] - 1
        device       = psi_prime.device

        demands  = instance["demands"]   # [mb, N+1]
        capacity = instance["capacity"]  # int or [mb] tensor

        # ── Precompute all T intermediate states from stored actions ──────
        # State at step t = result of applying actions[:, 0..t-1].
        # Uses in-place tensor writes into pre-allocated [mb, T, ...] buffers
        # — no env.step() calls, no visited.clone() per step.
        all_cur  = torch.zeros(mb, T, dtype=torch.long, device=device)           # [mb, T]
        all_used = torch.zeros(mb, T, device=device)                              # [mb, T]
        all_vis  = torch.zeros(mb, T, n_customers + 1, dtype=torch.bool, device=device)  # [mb, T, N+1]

        cur  = torch.zeros(mb, dtype=torch.long, device=device)
        used = torch.zeros(mb, device=device)
        vis  = torch.zeros(mb, n_customers + 1, dtype=torch.bool, device=device)

        for t in range(T):
            all_cur[:, t]  = cur               # current node BEFORE action t
            all_used[:, t] = used              # used capacity BEFORE action t
            all_vis[:, t]  = vis               # visited mask BEFORE action t (data copy)
            if t < T - 1:
                act_t    = actions[:, t]                                           # [mb]
                at_depot = (act_t == 0)
                dem_t    = demands.gather(1, act_t.unsqueeze(1)).squeeze(1)       # [mb]
                used     = torch.where(at_depot, torch.zeros_like(used), used + dem_t)
                vis.scatter_(1, act_t.unsqueeze(1), True)                         # in-place; all_vis[:, t] already saved
                vis[:, 0] = False                                                 # P7: depot never permanently visited
                cur      = act_t

        # ── Infeasibility mask [mb, T, N+1] ──────────────────────────────
        if isinstance(capacity, (int, float)):
            cap_float = float(capacity)
            remaining = cap_float - all_used                                      # [mb, T]
            cap_norm  = remaining / max(cap_float, 1e-8)                          # [mb, T]
        else:
            cap_t     = capacity.unsqueeze(-1)                                    # [mb, 1]
            remaining = cap_t - all_used                                          # [mb, T]
            cap_norm  = remaining / cap_t.clamp(min=1e-8)                        # [mb, T]

        # demands [mb, 1, N+1] > remaining [mb, T, 1] → [mb, T, N+1]
        exceeds  = demands.unsqueeze(1) > remaining.unsqueeze(-1)                 # [mb, T, N+1]
        mask_3d  = all_vis | exceeds                                              # [mb, T, N+1]  True=infeasible
        mask_3d[:, :, 0] = False                                                  # P3: depot always feasible
        # Empty-trip blocking: at depot + feasible customers exist → mask depot
        at_depot_3d = (all_cur == 0)                                              # [mb, T]
        has_cust_3d = (~mask_3d[:, :, 1:]).any(dim=-1)                           # [mb, T]
        mask_3d[:, :, 0] = mask_3d[:, :, 0] | (at_depot_3d & has_cust_3d)

        # ── ψ'_curr for all (instance, step) pairs ────────────────────────
        # psi_prime [mb, N+1, 2] gathered at current_node index → [mb, T, 2]
        cur_exp  = all_cur.unsqueeze(-1).expand(mb, T, 2)                         # [mb, T, 2]
        psi_curr = psi_prime.gather(1, cur_exp)                                   # [mb, T, 2]
        psi_curr = psi_curr.masked_fill(at_depot_3d.unsqueeze(-1), 0.0)          # zero at depot §4

        # ── Context [mb, T, 4] → query [mb, T, 2] ────────────────────────
        t_norm = (torch.arange(T, device=device).float() / max(n_customers, 1)   # [T]
                  ).unsqueeze(0).expand(mb, T)                                    # [mb, T]
        ctx    = torch.cat(                                                        # [mb, T, 4]
            [psi_curr, cap_norm.unsqueeze(-1), t_norm.unsqueeze(-1)], dim=-1
        )
        query  = self.decoder.context_query.Wq(ctx)                               # [mb, T, 2]

        # ── Hybrid scoring ────────────────────────────────────────────────
        # Context scores: q[b,t] · ψ'[b,j] for all j  →  [mb, T, N+1]
        context_scores = torch.einsum('btd,bnd->btn', query, psi_prime)

        # Interference computed ONCE per instance, broadcast over T steps
        interf = self.decoder.hybrid._eknn(psi_prime, knn_indices)               # [mb, N+1]
        interf = interf.unsqueeze(1)                                              # [mb, 1, N+1]

        scores     = context_scores + self.decoder.hybrid.lambda_param * interf  # [mb, T, N+1]
        scores     = scores.masked_fill(mask_3d, -1e9)                           # P2: mask BEFORE softmax
        log_probs_3d = F.log_softmax(scores, dim=-1)                             # [mb, T, N+1]

        # ── Gather log-probs for taken actions ────────────────────────────
        lp_new   = log_probs_3d.gather(2, actions.unsqueeze(-1)).squeeze(-1)     # [mb, T]

        # ── Entropy ───────────────────────────────────────────────────────
        ent_flat = Categorical(logits=log_probs_3d.reshape(mb * T, n_customers + 1)).entropy()  # [mb*T]
        entropy  = ent_flat.reshape(mb, T).mean(dim=1)                           # [mb]

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
