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
        feature_dim:  int   = 5,
        hidden_dim:   int   = 16,
        knn_k:        int   = 5,
        lambda_init:  float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.encoder = FullEncoder(feature_dim, 2, hidden_dim, knn_k)
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
        Roll out one complete episode.

        Returns:
            actions:      [B, T]
            log_probs:    [B, T]
            sum_log_prob: [B]
        """
        psi_prime, features, knn_indices = self.encoder(state)
        n_customers = psi_prime.shape[1] - 1

        all_actions, all_log_probs = [], []
        step = 0

        while not state["done"].all():
            log_probs, mask = self.decoder(
                state, psi_prime, knn_indices, step, n_customers
            )
            if deterministic:
                action = log_probs.argmax(dim=-1)                      # [B]
            else:
                action = Categorical(logits=log_probs).sample()        # [B]  §10

            log_p = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
            all_actions.append(action)
            all_log_probs.append(log_p)
            state = env.step(state, action)
            step += 1

        actions_t    = torch.stack(all_actions,   dim=1)               # [B, T]
        log_probs_t  = torch.stack(all_log_probs, dim=1)               # [B, T]
        return actions_t, log_probs_t, log_probs_t.sum(dim=1)

    def evaluate_actions(self, state: dict, env, actions: torch.Tensor):
        """Re-evaluate stored actions. Returns (log_probs [B,T], entropy [B])."""
        psi_prime, features, knn_indices = self.encoder(state)
        n_customers = psi_prime.shape[1] - 1
        T = actions.shape[1]
        lp_list, ent_list = [], []

        for t in range(T):
            log_probs, _ = self.decoder(
                state, psi_prime, knn_indices, t, n_customers
            )
            lp_list.append(log_probs.gather(1, actions[:, t:t+1]).squeeze(1))
            probs = log_probs.exp()
            ent_list.append(-(probs * log_probs).sum(dim=-1))
            state = env.step(state, actions[:, t])

        return torch.stack(lp_list, 1), torch.stack(ent_list, 1).mean(1)


# Backward-compat aliases
CVRPPolicy = QAPPolicy


class CVRPCritic(nn.Module):
    """Standalone critic for PPO trainer compat. 2→64→1 (~257 params)."""

    def __init__(self, feature_dim: int = 5, hidden_dim: int = 16, **kwargs):
        super().__init__()
        self.encoder = FullEncoder(feature_dim, 2, hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, state: dict, step: int = 0, n_customers: int = 0):
        psi_prime, _, _ = self.encoder(state)                          # [B, N+1, 2]
        pooled = psi_prime.mean(dim=1)                                 # [B, 2]
        return self.value_head(pooled).squeeze(-1)                     # [B]
