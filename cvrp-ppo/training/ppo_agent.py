"""
training/ppo_agent.py
======================
Step 7 — PPO Update.

Implements the full PPO training loop:

    Collect rollout  →  compute GAE  →  ppo_epochs × minibatch updates
        L = E[ min( rₜ·Aₜ,  clip(rₜ, 1−ε, 1+ε)·Aₜ ) ]
        where rₜ = π_new(aₜ|sₜ) / π_old(aₜ|sₜ)
        R  = −Total Distance
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# local imports — add repo root to path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.rollout_buffer import RolloutBuffer
from utils.metrics           import compute_metrics
from utils.logger            import Logger
from utils.checkpoint        import save_checkpoint, load_checkpoint


class PPOTrainer:
    """
    PPO trainer for CVRP.

    Args:
        policy:          CVRPPolicy  (actor)
        critic:          CVRPCritic  (value network)
        env:             CVRPEnv
        generator:       CVRPGenerator
        clip_epsilon:    PPO clip range ε            (default 0.2)
        entropy_coef:    entropy bonus weight        (default 0.01)
        value_coef:      value loss weight           (default 0.5)
        max_grad_norm:   gradient clipping           (default 1.0)
        ppo_epochs:      gradient steps per rollout  (default 3)
        n_minibatches:   minibatches per PPO epoch   (default 8)
        gamma:           reward discount             (default 0.99)
        gae_lambda:      GAE λ                       (default 0.95)
        lr:              learning rate               (default 1e-4)
        batch_size:      instances per rollout       (default 64)
        device:          torch device string         (default "cpu")
        log_dir:         directory for logs/ckpts    (default "outputs/")
    """

    def __init__(
        self,
        policy,
        critic,
        env,
        generator,
        clip_epsilon:  float = 0.2,
        entropy_coef:  float = 0.01,
        value_coef:    float = 0.5,
        max_grad_norm: float = 1.0,
        ppo_epochs:    int   = 3,
        n_minibatches: int   = 8,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        lr:            float = 1e-4,
        batch_size:    int   = 64,
        device:        str   = "cpu",
        log_dir:       str   = "outputs",
    ):
        self.policy        = policy.to(device)
        self.critic        = critic.to(device)
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

        # Combine all trainable parameters into one optimizer
        all_params = list(policy.parameters()) + list(critic.parameters())
        self.optimizer = Adam(all_params, lr=lr)

        self.buffer = RolloutBuffer(device=device)
        self.logger = Logger(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Rollout collection
    # ─────────────────────────────────────────────────────────────────────────

    def collect_rollout(self) -> float:
        """
        Run policy in env for one episode (B instances in parallel).

        Returns:
            mean_reward: float  — mean total distance (positive) for logging
        """
        instance = self.generator.generate(self.batch_size, device=self.device)
        state    = self.env.reset(instance)
        n_cust   = self.generator.num_loc

        self.buffer.clear()
        self.policy.eval()
        self.critic.eval()

        all_actions = []
        step = 0

        with torch.no_grad():
            # Encode once — cached for all decode steps
            psi_prime, _, knn_indices = self.policy.encoder(state)
            value = self.critic(state, step, n_cust)

            while not state["done"].all():
                # Policy step (stochastic)
                log_probs, mask = self.policy.decoder(
                    state, psi_prime, knn_indices, step, n_cust
                )

                dist    = torch.distributions.Categorical(logits=log_probs)
                action  = dist.sample()
                log_p   = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)

                self.buffer.add_step(log_p, value, action)
                all_actions.append(action)
                state = self.env.step(state, action)
                step += 1

        # Episode reward R = −total_distance
        actions_t = torch.stack(all_actions, dim=1)
        reward    = self.env.get_reward(state, actions_t)      # [B], negative

        self.buffer.set_rewards(reward)
        self.buffer.compute_gae(
            last_value = torch.zeros(self.batch_size, device=self.device),
            gamma      = self.gamma,
            gae_lambda = self.gae_lambda,
        )

        # Store initial state and actions for re-evaluation
        self._last_instance = instance
        self._last_actions  = actions_t

        mean_reward = reward.mean().item()
        return mean_reward

    # ─────────────────────────────────────────────────────────────────────────
    # PPO update
    # ─────────────────────────────────────────────────────────────────────────

    def update(self) -> dict:
        """
        Perform ppo_epochs × n_minibatches gradient updates on the stored rollout.

        Returns:
            loss_dict: dict with keys policy_loss, value_loss, entropy, total_loss
        """
        self.policy.train()
        self.critic.train()

        total_ploss, total_vloss, total_ent = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for (idx, lp_old, acts, advs, rets, vals_old) in \
                    self.buffer.get_minibatches(self.n_minibatches):

                # Re-evaluate with current policy — use idx to match correct instances
                init_instance = {
                    k: v[idx] if isinstance(v, torch.Tensor) else v
                    for k, v in self._last_instance.items()
                }
                init_state = self.env.reset(init_instance)

                lp_new, entropy = self.policy.evaluate_actions(
                    init_state, self.env, acts
                )

                # ── PPO clipped objective ─────────────────────────────────
                # rₜ = exp(log π_new − log π_old)
                ratio   = (lp_new - lp_old).exp()                 # [mb, T]
                adv_t   = advs                                      # [mb, T]

                surr1   = ratio * adv_t
                surr2   = ratio.clamp(
                    1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * adv_t
                p_loss  = -torch.min(surr1, surr2).mean()

                # ── Value loss ────────────────────────────────────────────
                # Re-compute values for this minibatch
                state_mb = self.env.reset(init_instance)
                n_cust   = self.generator.num_loc
                v_new    = self.critic(state_mb, 0, n_cust)        # [mb]
                v_loss   = ((v_new - rets[:, 0]) ** 2).mean()

                # ── Entropy bonus ─────────────────────────────────────────
                ent_loss = -entropy.mean()

                # ── Total loss ────────────────────────────────────────────
                loss = p_loss + self.value_coef * v_loss + self.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_ploss += p_loss.item()
                total_vloss += v_loss.item()
                total_ent   += (-ent_loss.item())
                n_updates   += 1

        return {
            "policy_loss": total_ploss / n_updates,
            "value_loss":  total_vloss / n_updates,
            "entropy":     total_ent   / n_updates,
            "total_loss":  (total_ploss + total_vloss + total_ent) / n_updates,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        n_iterations:     int,
        val_dataset       = None,
        eval_every:       int = 10,
        checkpoint_every: int = 50,
        resume_from:      str = None,
        on_iteration_end  = None,
    ):
        """
        Full PPO training loop.

        Args:
            n_iterations:     total number of rollout → update cycles
            val_dataset:      optional validation dataset for periodic eval
            eval_every:       evaluate every N iterations
            checkpoint_every: save checkpoint every N iterations
            resume_from:      path to checkpoint to resume from

        Returns:
            history: dict of lists (reward, losses per iteration)
        """
        start_iter = 0
        if resume_from:
            start_iter = load_checkpoint(
                resume_from, self.policy, self.critic, self.optimizer
            )
            print(f"[Resume] Starting from iteration {start_iter}")

        history = {"reward": [], "policy_loss": [], "value_loss": [], "entropy": []}

        for it in range(start_iter, n_iterations):
            t0 = time.time()

            # Free unused GPU memory at start of each iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Collect rollout
            mean_reward = self.collect_rollout()

            # PPO update
            losses = self.update()

            elapsed = time.time() - t0
            history["reward"].append(mean_reward)
            history["policy_loss"].append(losses["policy_loss"])
            history["value_loss"].append(losses["value_loss"])
            history["entropy"].append(losses["entropy"])

            # Log
            self.logger.log_scalars(
                {"reward": mean_reward, **losses}, step=it
            )
            print(
                f"[{it+1:04d}/{n_iterations}] "
                f"reward={mean_reward:.4f}  "
                f"p_loss={losses['policy_loss']:.4f}  "
                f"v_loss={losses['value_loss']:.4f}  "
                f"ent={losses['entropy']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            if on_iteration_end is not None:
                on_iteration_end(it + 1, mean_reward, losses)

            # Validate
            if val_dataset is not None and (it + 1) % eval_every == 0:
                val_reward = self._validate(val_dataset)
                print(f"  → Val reward: {val_reward:.4f}")
                self.logger.log_scalars({"val_reward": val_reward}, step=it)

            # Checkpoint
            if (it + 1) % checkpoint_every == 0:
                path = os.path.join(self.log_dir, f"ckpt_{it+1:04d}.pt")
                save_checkpoint(
                    self.policy, self.critic, self.optimizer, it + 1,
                    {"reward": mean_reward}, path
                )

        self.logger.close()
        return history

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self, dataset, n_samples: int = 256) -> float:
        """Greedy decoding on val instances, returns mean reward."""
        self.policy.eval()
        instance = {k: v[:n_samples].to(self.device) for k, v in dataset.data.items()}
        state    = self.env.reset(instance)

        with torch.no_grad():
            actions, _, _ = self.policy(state, self.env, deterministic=True)

        reward = self.env.get_reward(state, actions)
        return reward.mean().item()
