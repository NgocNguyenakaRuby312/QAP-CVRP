"""
decoder/qap_decoder.py
=======================
Autoregressive decoder: repeats Steps 5 + 6 until all N customers visited.

    psi_curr=0 when at depot → context → query → score → mask → sample → update

Change 1 + Change 2 (May 2026):
    - context_query now returns (query, current_coords)
    - hybrid scoring now receives current_coords and all_coords
    - context_dim defaults to 8 (D+4 where D=4)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, Optional

from .context_query  import ContextAndQuery
from .hybrid_scoring import HybridScoring


class QAPDecoder(nn.Module):
    """
    Full autoregressive decoder for constructive CVRP.

    Args:
        context_dim:  context vector size (default 6 — includes x_curr, y_curr)
        embed_dim:    amplitude / query space size (default 2)
        lambda_init:  initial hybrid weight λ (default 0.1)
        mu_init:      initial distance penalty weight μ (default 0.5)
    """

    def __init__(self, context_dim: int = 8, embed_dim: int = 4,
                 lambda_init: float = 0.1, mu_init: float = 0.5):
        super().__init__()
        self.context_query = ContextAndQuery(context_dim, embed_dim)
        self.hybrid        = HybridScoring(lambda_init, mu_init)

    # ── single decode step ───────────────────────────────────────────

    def forward(
        self,
        state:       dict,
        psi_prime:   torch.Tensor,    # [B, N+1, D]  static, fixed for all steps
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

        # Change 1+2: context_query returns (query, current_coords)
        query, current_coords = self.context_query(                    # [B,2], [B,2]
            state, psi_prime, step, n_customers
        )

        # scoring with all 4 terms (demands/capacity for nu term)
        log_probs = self.hybrid(
            query, psi_prime, knn_indices, mask,
            current_coords, state["coords"],
            demands=state.get("demands"),
            capacity=state.get("capacity", 1.0),
        )                                                              # [B, N+1]
        return log_probs, mask

    # ── full-episode rollout ─────────────────────────────────────────

    def rollout(
        self,
        psi_prime:   torch.Tensor,    # [B, N+1, D]  fixed for all steps
        env_state:   dict,
        knn_indices: torch.Tensor,    # [B, N+1, k]  precomputed — does not change
        env,
        greedy:      bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressive loop until all customers visited.

        psi_prime is STATIC — computed once before calling rollout().
        No per-step re-encoding.

        Args:
            psi_prime:   [B, N+1, D]  encoder output (fixed)
            env_state:   dict         initial environment state
            knn_indices: [B, N+1, k]  precomputed spatial kNN
            env:         CVRPEnv      dict-based step interface
            greedy:      True=argmax, False=Categorical sample

        Returns:
            actions:     [B, T]  selected nodes
            log_probs:   [B, T]  per-step log-probabilities
            tour_length: [B]    total Euclidean tour distance (positive)
        """
        n_customers = psi_prime.shape[1] - 1
        state = env_state
        device = psi_prime.device
        B = psi_prime.shape[0]

        all_actions   = []
        all_log_probs = []

        max_steps = 3 * n_customers + 1
        for step in range(max_steps):
            if state["done"].all():
                break

            log_probs, mask = self.forward(
                state, psi_prime, knn_indices, step, n_customers
            )                                                          # [B, N+1]

            if greedy:
                action = log_probs.argmax(dim=-1)                      # [B]
            else:
                dist   = Categorical(logits=log_probs)                 # §10
                action = dist.sample()                                 # [B]

            log_p = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)  # [B]

            all_actions.append(action)
            all_log_probs.append(log_p)

            state = env.step(state, action)

        actions_t   = torch.stack(all_actions,   dim=1)                # [B, T]
        log_probs_t = torch.stack(all_log_probs, dim=1)                # [B, T]
        tour_length = self._tour_length(state["coords"], actions_t)    # [B]

        return actions_t, log_probs_t, tour_length

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_mask(state: dict) -> torch.Tensor:
        """Infeasibility mask: True = cannot select. Depot feasible unless empty trip."""
        if "action_mask" in state:
            mask = ~state["action_mask"]                               # [B, N+1]
        else:
            visited  = state["visited"]
            demands  = state["demands"]
            cap  = state["capacity"]
            used = state["used_capacity"]
            if isinstance(cap, (int, float)):
                remaining = cap - used
            else:
                remaining = cap - used
            exceeds = demands > remaining.unsqueeze(-1)                # [B, N+1]
            mask = visited | exceeds
            mask[:, 0] = False                                         # P3: depot feasible
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
