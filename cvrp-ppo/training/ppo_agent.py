"""
training/ppo_agent.py
======================
Step 7 — PPO Update.

Implements the full PPO training loop with **greedy rollout baseline**:

    Collect sampled + greedy rollout  →  advantage = R_sample − R_greedy
    →  ppo_epochs × minibatch updates with clipped objective

FIX LOG:
  v3:
    - CRITICAL FIX: psi_prime DETACHED before critic head in update()
      Prevents value loss gradient from corrupting the shared encoder.
    - lr: restored to 1e-4 (thesis spec §3.X.8)
    - entropy_coef: restored to 0.01 (thesis spec c2)
    - value_coef: now passed explicitly, default 0.5 (thesis spec c1)
    - LOGGING: update() now returns a full diagnostic dict (15+ fields).
  v4:
    - eta_min raised 1e-6 → 1e-5 in CosineAnnealingLR.
  v5 (Changes 1+2):
    - update() diagnostic dict now also returns mu_val — the current value
      of the learnable distance penalty scalar μ (Change 1, §3.3.4).
    - collect_rollout() uses static encoder: no per-step re-encoding.
    - Encoder is STATIC: psi_prime computed once, fixed for all decode steps.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.rollout_buffer import RolloutBuffer
from utils.metrics           import compute_metrics
from utils.logger            import Logger
from utils.checkpoint        import save_checkpoint, load_checkpoint


class PPOTrainer:
    """
    PPO trainer for CVRP with greedy rollout baseline (AM/POMO-style).

    collect_rollout() stores rich rollout stats in self._rollout_stats.
    update() returns a 16-field diagnostic dict covering every failure mode:
        policy_loss, value_loss, entropy        — three separate loss terms
        grad_norm                               — detect explosion
        clip_fraction                           — detect over-clipping
        ratio_mean                              — PPO ratio health
        adv_mean, adv_std                       — learning signal quality
        train_tour, greedy_tour, improvement    — policy vs baseline
        lambda_val                              — is λ learning?
        mu_val                                  — is μ learning? (Change 1)
        lr                                      — cosine schedule tracking
    """

    def __init__(
        self,
        policy,
        critic        = None,
        env           = None,
        generator     = None,
        clip_epsilon:  float = 0.2,
        entropy_coef:  float = 0.01,   # thesis spec c2
        value_coef:    float = 0.5,    # thesis spec c1
        max_grad_norm: float = 1.0,
        ppo_epochs:    int   = 3,
        n_minibatches: int   = 8,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        lr:            float = 1e-4,   # thesis spec
        batch_size:    int   = 256,
        total_steps:   int   = 600_000,
        device:        str   = "cpu",
        log_dir:       str   = "outputs",
    ):
        self.policy        = policy.to(device)
        self.env           = env
        self.generator     = generator
        self.clip_epsilon  = clip_epsilon
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs    = ppo_epochs
        self.n_minibatches = n_minibatches
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.batch_size    = batch_size
        self.device        = device
        self.log_dir       = log_dir

        # Fix 1: single param group — no weight_decay on μ (let it find natural value)
        # μ clamp [0,10] prevents explosion; weight_decay was crushing it to 0.09
        self.optimizer = Adam(policy.parameters(), lr=lr)
        # Fix 3: CosineAnnealingWarmRestarts — periodic LR resets to escape local minima
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-5  # restarts at ep 50, 150
        )

        self.buffer = RolloutBuffer(device=device)
        self.logger = Logger(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Populated by collect_rollout(), read by update() and train scripts
        self._rollout_stats   = {}
        self._last_instance   = None
        self._last_actions    = None
        self._last_knn        = None

    # ─────────────────────────────────────────────────────────────────────────
    # Rollout collection
    # ─────────────────────────────────────────────────────────────────────────

    def collect_rollout(self) -> float:
        """
        Collect one rollout (sampled + greedy baseline).

        Encoder is STATIC: psi_prime computed once before decoding loop,
        fixed for all decode steps. No per-step re-encoding.

        Stores diagnostic stats in self._rollout_stats.
        Returns mean sampled tour length (positive distance).
        """
        instance = self.generator.generate(self.batch_size, device=self.device)
        n_cust   = self.generator.num_loc

        self.buffer.clear()
        self.policy.eval()

        with torch.no_grad():
            state_sample = self.env.reset(instance)
            # Encode ONCE — psi_prime fixed for all decode steps
            psi_prime, _, knn_indices = self.policy.encoder(state_sample)  # [B,N+1,D]
            self._last_knn = knn_indices

            # ── Pass 1: Sampled rollout ────────────────────────────────
            # psi_prime is static — same for every decode step
            all_actions       = []
            state             = state_sample

            max_steps = 3 * n_cust + 1
            for step in range(max_steps):
                if state["done"].all():
                    break

                log_probs, _ = self.policy.decoder(
                    state, psi_prime, knn_indices, step, n_cust    # [B, N+1]
                )
                dist   = torch.distributions.Categorical(logits=log_probs)
                action = dist.sample()                                 # [B]
                log_p  = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)

                self.buffer.add_step(log_p, action)
                all_actions.append(action)
                state = self.env.step(state, action)

            actions_t     = torch.stack(all_actions, dim=1)            # [B, T]
            reward_sample = self.env.get_reward(state, actions_t)      # [B] negative

            # ── Pass 2: Greedy rollout (baseline) ──────────────────────
            # Static psi_prime — no encoder arg needed
            state_greedy = self.env.reset(instance)
            _, _, tour_len_greedy = self.policy.decoder.rollout(
                psi_prime, state_greedy, knn_indices, self.env,
                greedy=True,
            )
            reward_greedy = -tour_len_greedy                           # [B] negative

        advantage = reward_sample - reward_greedy                      # [B]
        self.buffer.set_advantage(advantage)
        self.buffer.set_returns(reward_sample)

        self._last_instance = instance
        self._last_actions  = actions_t

        train_tour  = (-reward_sample).mean().item()
        greedy_tour = tour_len_greedy.mean().item()
        self._rollout_stats = {
            "train_tour":  train_tour,
            "greedy_tour": greedy_tour,
            "improvement": greedy_tour - train_tour,
            "adv_raw_std": advantage.std().item(),
        }

        return reward_sample.mean().item()

    # ─────────────────────────────────────────────────────────────────────────
    # PPO update — returns full 16-field diagnostic dict
    # ─────────────────────────────────────────────────────────────────────────

    def update(self) -> dict:
        """
        Perform ppo_epochs × n_minibatches gradient updates.

        CRITICAL: psi_prime DETACHED before critic head to stop value gradient
        from corrupting the shared encoder.

        Encoder is STATIC: feature_builder(init_instance) called once per
        minibatch with no current_node_coords. evaluate_actions() in
        qap_policy.py uses the resulting static psi_prime_mb, broadcast
        across all T steps.

        Returns a comprehensive diagnostic dict with 16 fields:
            policy_loss, value_loss, entropy, grad_norm, clip_fraction,
            ratio_mean, adv_mean, adv_std, train_tour, greedy_tour,
            improvement, lambda_val, mu_val (Change 1), lr
        """
        self.policy.train()

        adv_raw = self.buffer.advantages                                # [B]
        adv_std = adv_raw.std()
        if adv_std > 1e-8:
            self.buffer.advantages = (adv_raw - adv_raw.mean()) / (adv_std + 1e-8)

        total_ploss = total_vloss = total_ent = 0.0
        total_gnorm = total_clip  = total_ratio = 0.0
        n_updates   = 0

        for _ in range(self.ppo_epochs):
            for (idx, lp_old, acts, advs, rets) in \
                    self.buffer.get_minibatches(self.n_minibatches):

                init_instance = {
                    k: v[idx] if isinstance(v, torch.Tensor) else v
                    for k, v in self._last_instance.items()
                }
                # Static encode: feature_builder returns [mb, N+1, 5]
                features_mb  = self.policy.encoder.feature_builder(init_instance)
                psi_prime_mb = self.policy.encoder.qap_encoder(features_mb)  # [mb,N+1,D]
                knn_mb       = self._last_knn[idx]                            # [mb,N+1,k]

                lp_new, entropy = self.policy.evaluate_actions(
                    init_instance, acts, psi_prime_mb, knn_mb
                )

                # ── PPO clipped objective ──────────────────────────────
                ratio  = (lp_new - lp_old).exp()                       # [mb, T]
                adv_t  = advs.unsqueeze(1)                              # [mb, 1]
                surr1  = ratio * adv_t
                surr2  = ratio.clamp(
                    1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * adv_t
                p_loss = -torch.min(surr1, surr2).mean()

                clipped = ((ratio < 1 - self.clip_epsilon) |
                           (ratio > 1 + self.clip_epsilon)).float().mean().item()

                # ── Value loss — DETACHED (encoder protected) ──────────
                v_new  = self.policy.get_value(psi_prime_mb.detach())   # [mb]
                v_loss = ((v_new - rets) ** 2).mean()

                ent_loss = -entropy.mean()

                loss = p_loss + self.value_coef * v_loss + self.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()

                raw_gnorm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                ).item()

                self.optimizer.step()
                self.scheduler.step()

                total_ploss += p_loss.item()
                total_vloss += v_loss.item()
                total_ent   += (-ent_loss.item())
                total_gnorm += raw_gnorm
                total_clip  += clipped
                total_ratio += ratio.mean().item()
                n_updates   += 1

        n = n_updates
        return {
            # ── Three separate losses ─────────────────────────────────
            "policy_loss":    total_ploss / n,
            "value_loss":     total_vloss / n,
            "entropy":        total_ent   / n,
            # ── Gradient health ───────────────────────────────────────
            "grad_norm":      total_gnorm / n,
            # ── PPO ratio health ──────────────────────────────────────
            "clip_fraction":  total_clip  / n,
            "ratio_mean":     total_ratio / n,
            # ── Advantage / learning signal quality ───────────────────
            "adv_mean":  self.buffer.advantages.mean().item(),
            "adv_std":   self._rollout_stats["adv_raw_std"],
            # ── Policy vs baseline ─────────────────────────────────────
            "train_tour":    self._rollout_stats["train_tour"],
            "greedy_tour":   self._rollout_stats["greedy_tour"],
            "improvement":   self._rollout_stats["improvement"],
            # ── Model state ───────────────────────────────────────────
            "lambda_val":    self.policy.decoder.hybrid.lambda_param.item(),
            "mu_val":        self.policy.decoder.hybrid.mu_param.item(),
            "nu_val":        self.policy.decoder.hybrid.nu_param.item(),
            "lr":            self.optimizer.param_groups[0]["lr"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self, dataset, n_samples: int = 256) -> float:
        self.policy.eval()
        instance = {k: v[:n_samples].to(self.device) for k, v in dataset.data.items()}
        state    = self.env.reset(instance)
        with torch.no_grad():
            actions, _, _ = self.policy(state, self.env, deterministic=True)
        reward = self.env.get_reward(state, actions)
        return reward.mean().item()

    @property
    def critic(self):
        return self.policy.critic_head
