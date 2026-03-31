"""
training/rollout_buffer.py
===========================
Stores trajectories collected from the environment during PPO rollout and
computes Generalised Advantage Estimates (GAE).
"""

import torch


class RolloutBuffer:
    """
    Fixed-size circular buffer for one PPO iteration of experience.

    Stores per-step tuples:
        (log_prob, value, reward, done, action)

    After a complete episode, call compute_gae() to fill advantages and
    value-targets, then iterate minibatches via get_minibatches().

    Args:
        device: torch device
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.clear()

    # ── accumulate ───────────────────────────────────────────────────────────

    def clear(self):
        self.log_probs  = []   # list of [B] tensors, one per step
        self.values     = []
        self.rewards    = []   # filled after episode end
        self.dones      = []
        self.actions    = []   # list of [B] tensors

    def add_step(
        self,
        log_prob: torch.Tensor,   # [B]
        value:    torch.Tensor,   # [B]
        action:   torch.Tensor,   # [B]
    ):
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.actions.append(action.detach())

    def set_rewards(self, rewards: torch.Tensor):
        """
        Set the episode reward (scalar per instance, applied to last step).
        Called once after env.get_reward() at the end of a rollout.

        Args:
            rewards: [B]  reward = −total_distance
        """
        T = len(self.log_probs)
        B = rewards.shape[0]
        self.rewards = [torch.zeros(B, device=self.device)] * (T - 1)
        self.rewards.append(rewards)    # reward arrives at the final step
        self.dones   = [torch.zeros(B, dtype=torch.bool, device=self.device)] * (T - 1)
        self.dones.append(torch.ones(B, dtype=torch.bool, device=self.device))

    # ── advantage computation (GAE) ──────────────────────────────────────────

    def compute_gae(
        self,
        last_value: torch.Tensor,    # [B]  V(s_T) = 0 for terminal
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Generalised Advantage Estimation:
            δₜ  = rₜ + γ·V(sₜ₊₁) − V(sₜ)
            Aₜ  = Σ_{l≥0} (γλ)^l · δₜ₊ₗ

        Stores advantages and returns as tensors [T, B].
        """
        T = len(self.values)
        values_ext = self.values + [last_value]    # V(s₀)…V(s_T)

        advantages = []
        gae = torch.zeros_like(last_value)

        for t in reversed(range(T)):
            next_val   = values_ext[t + 1] * (~self.dones[t]).float()
            delta      = self.rewards[t] + gamma * next_val - self.values[t]
            gae        = delta + gamma * gae_lambda * gae * (~self.dones[t]).float()
            advantages.insert(0, gae.clone())

        self.advantages = torch.stack(advantages, dim=0)   # [T, B]
        self.returns    = self.advantages + torch.stack(self.values, dim=0)

        # Normalise advantages
        adv_flat = self.advantages
        self.advantages = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    # ── minibatch iterator ───────────────────────────────────────────────────

    def get_minibatches(self, n_minibatches: int):
        """
        Yield n_minibatches random sub-batches of stored experience.

        Each yield: (log_probs_old, actions_all, advantages, returns, values_old)
            shapes: [mb_size, T]  where mb_size = B // n_minibatches
        """
        T, B = self.advantages.shape
        perm = torch.randperm(B, device=self.device)
        mb   = B // n_minibatches

        log_probs_t = torch.stack(self.log_probs, dim=0)   # [T, B]
        actions_t   = torch.stack(self.actions,   dim=0)   # [T, B]
        values_t    = torch.stack(self.values,    dim=0)   # [T, B]

        for i in range(n_minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            yield (
                idx,                        # [mb]   instance indices
                log_probs_t[:, idx].T,      # [mb, T]
                actions_t[:, idx].T,        # [mb, T]
                self.advantages[:, idx].T,  # [mb, T]
                self.returns[:, idx].T,     # [mb, T]
                values_t[:, idx].T,         # [mb, T]
            )
