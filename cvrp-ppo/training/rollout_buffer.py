"""
training/rollout_buffer.py
===========================
Stores trajectories collected from the environment during PPO rollout.

Architecture note (BUG C2): The QAP encoder output psi_prime is static per
instance — it is cached once before decoding and never updated.  Therefore
V(sₜ) = V for every step t, making step-level GAE degenerate:

    δₜ = rₜ + γ·V(sₜ₊₁) − V(sₜ)  →  rₜ + (γ−1)·V  ≈  rₜ − 0.01·V

The advantages would be dominated by the constant −0.01·V rather than
actual action credit.

Correct design: **instance-level REINFORCE baseline**

    advantage = reward − V(instance)      [B]  — one number per instance
    returns   = reward                    [B]  — total episode return

The same advantage is broadcast across all T action steps in the PPO loss.
"""

import torch


class RolloutBuffer:
    """
    Buffer for one PPO iteration with an instance-level REINFORCE baseline.

    Stores per-step tuples (log_prob, action).  After the episode completes,
    call set_advantage() and set_returns() once, then iterate minibatches
    via get_minibatches().

    Args:
        device: torch device
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.clear()

    # ── accumulate ───────────────────────────────────────────────────────────

    def clear(self):
        self.log_probs  = []    # list of [B] tensors, one per decode step
        self.actions    = []    # list of [B] tensors, one per decode step
        self.advantages = None  # [B]  set after episode
        self.returns    = None  # [B]  set after episode

    def add_step(
        self,
        log_prob: torch.Tensor,   # [B]
        action:   torch.Tensor,   # [B]
    ):
        self.log_probs.append(log_prob.detach())
        self.actions.append(action.detach())

    # ── instance-level advantage ─────────────────────────────────────────────

    def set_advantage(self, advantage: torch.Tensor):
        """
        Store instance-level advantage = reward − baseline.

        Args:
            advantage: [B]  one value per instance, broadcast to all T steps
        """
        self.advantages = advantage.detach()

    def set_returns(self, returns: torch.Tensor):
        """
        Store total episode return per instance (used as value-loss target).

        Args:
            returns: [B]  total episode return = −total_distance
        """
        self.returns = returns.detach()

    # ── minibatch iterator ───────────────────────────────────────────────────

    def get_minibatches(self, n_minibatches: int):
        """
        Yield n_minibatches random sub-batches of stored experience.

        Each yield: (idx, log_probs_old, actions_all, advantages, returns)
            idx:           [mb]     instance indices into _last_instance
            log_probs_old: [mb, T]
            actions_all:   [mb, T]
            advantages:    [mb]     same advantage applied to every step t
            returns:       [mb]     total episode return (value-loss target)
        """
        B    = self.advantages.shape[0]
        perm = torch.randperm(B, device=self.device)
        # FIXED: clamp n_minibatches so mb >= 1 (avoids 0-element batches when B < n_minibatches)
        n_minibatches = min(n_minibatches, B)
        mb   = B // n_minibatches

        log_probs_t = torch.stack(self.log_probs, dim=0)   # [T, B]
        actions_t   = torch.stack(self.actions,   dim=0)   # [T, B]

        for i in range(n_minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            yield (
                idx,                        # [mb]
                log_probs_t[:, idx].T,      # [mb, T]
                actions_t[:, idx].T,        # [mb, T]
                self.advantages[idx],       # [mb]
                self.returns[idx],          # [mb]
            )
