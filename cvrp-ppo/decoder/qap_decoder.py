"""
decoder/qap_decoder.py
=======================
Autoregressive decoder: repeats Steps 5 + 6 until all N customers visited.

    psi_curr=0 when at depot → context → query → score → mask → sample → update
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple

from .context_query  import ContextAndQuery
from .hybrid_scoring import HybridScoring


class QAPDecoder(nn.Module):
    """
    Full autoregressive decoder for constructive CVRP.

    Args:
        context_dim:  context vector size (default 4)
        embed_dim:    amplitude / query space size (default 2)
        lambda_init:  initial hybrid weight λ (default 0.1)
    """

    def __init__(self, context_dim: int = 4, embed_dim: int = 2,
                 lambda_init: float = 0.1):
        super().__init__()
        self.context_query = ContextAndQuery(context_dim, embed_dim)
        self.hybrid        = HybridScoring(lambda_init)

    # ── single decode step ───────────────────────────────────────────

    def forward(
        self,
        state:       dict,
        psi_prime:   torch.Tensor,
        knn_indices: torch.Tensor,
        step:        int,
        n_customers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One decode step.

        Returns:
            log_probs: [B, N+1]  log-softmax scores
            mask:      [B, N+1]  True = infeasible
        """
        mask  = self._get_mask(state)                                  # [B, N+1]
        query = self.context_query(state, psi_prime, step, n_customers)  # [B, 2]
        log_probs = self.hybrid(query, psi_prime, knn_indices, mask)   # [B, N+1]
        return log_probs, mask

    # ── full-episode rollout ─────────────────────────────────────────

    def rollout(
        self,
        psi_prime:   torch.Tensor,
        env_state:   dict,
        knn_indices: torch.Tensor,
        env,
        greedy:      bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressive loop until all customers visited.

        At depot: psi_curr = zero vector [B, 2].                       # §4

        Args:
            psi_prime:   [B, N+1, 2]  encoder output (cached)
            env_state:   dict         initial environment state
            knn_indices: [B, N+1, k]  precomputed spatial kNN
            env:         CVRPEnv      dict-based step interface
            greedy:      True=argmax, False=Categorical sample         # §10

        Returns:
            actions:     [B, T]  selected nodes
            log_probs:   [B, T]  per-step log-probabilities
            tour_length: [B]    total Euclidean tour distance (positive)
        """
        n_customers = psi_prime.shape[1] - 1
        state = env_state

        all_actions   = []
        all_log_probs = []
        step = 0

        while not state["done"].all():
            log_probs, mask = self.forward(
                state, psi_prime, knn_indices, step, n_customers
            )                                                          # [B, N+1]

            if greedy:
                action = log_probs.argmax(dim=-1)                      # [B]
            else:
                dist   = Categorical(logits=log_probs)                 # §10: Categorical
                action = dist.sample()                                 # [B]

            log_p = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)  # [B]

            all_actions.append(action)
            all_log_probs.append(log_p)

            state = env.step(state, action)
            step += 1

        actions_t   = torch.stack(all_actions,   dim=1)                # [B, T]
        log_probs_t = torch.stack(all_log_probs, dim=1)                # [B, T]

        # Tour length: depot → route → depot
        tour_length = self._tour_length(state["coords"], actions_t)    # [B]

        return actions_t, log_probs_t, tour_length

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_mask(state: dict) -> torch.Tensor:
        """Infeasibility mask: True = cannot select. Depot feasible unless empty trip."""
        if "action_mask" in state:
            # action_mask True=feasible → invert to True=infeasible
            mask = ~state["action_mask"]                               # [B, N+1]
        else:
            visited  = state["visited"]
            demands  = state["demands"]
            cap = state["capacity"]
            used = state["used_capacity"]
            if isinstance(cap, (int, float)):
                remaining = cap - used
            else:
                remaining = cap - used
            exceeds = demands > remaining.unsqueeze(-1)                # [B, N+1]
            mask = visited | exceeds
            mask[:, 0] = False                                         # P3: depot feasible by default
            # Block empty trips: at depot + customers available → mask depot
            at_depot = (state["current_node"] == 0)                    # [B]
            has_cust = (~mask[:, 1:]).any(dim=1)                       # [B]
            mask[:, 0] = mask[:, 0] | (at_depot & has_cust)
        return mask

    @staticmethod
    def _tour_length(coords: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Total Euclidean tour distance depot→route→depot."""
        B, T = actions.shape
        idx   = actions.unsqueeze(-1).expand(B, T, 2)                  # [B, T, 2]
        route = coords.gather(1, idx)                                  # [B, T, 2]
        depot = coords[:, 0:1, :]                                      # [B, 1, 2]
        full  = torch.cat([depot, route, depot], dim=1)                # [B, T+2, 2]
        return (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)  # [B]
