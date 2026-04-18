"""
options.py
==========
argparse with ALL hyperparameters from CLAUDE.md §6.
"""

import argparse
import torch

CAPACITY_MAP = {10: 20, 20: 30, 50: 40, 100: 50}


def get_options(args=None):
    """Parse command-line arguments. All defaults match CLAUDE.md §6."""
    parser = argparse.ArgumentParser(description="QAP-DRL for CVRP")

    # ── Problem ──────────────────────────────────────────────────────
    parser.add_argument("--graph_size",   type=int,   default=20, choices=[10, 20, 50, 100])
    parser.add_argument("--capacity",     type=int,   default=None)

    # ── Architecture ─────────────────────────────────────────────────
    parser.add_argument("--embedding_dim", type=int,   default=2)
    parser.add_argument("--hidden_dim",    type=int,   default=16)
    parser.add_argument("--knn_k",         type=int,   default=5)
    parser.add_argument("--lambda_init",   type=float, default=0.1)

    # ── PPO ──────────────────────────────────────────────────────────
    parser.add_argument("--K_epochs",      type=int,   default=3)
    parser.add_argument("--eps_clip",      type=float, default=0.2)
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--gae_lambda",    type=float, default=0.95)
    parser.add_argument("--c1",            type=float, default=0.5)
    parser.add_argument("--c2",            type=float, default=0.05,
                        help="Entropy bonus weight (FIXED: was 0.01)")
    parser.add_argument("--n_minibatches", type=int,   default=8,
                        help="Minibatches per PPO epoch")

    # ── Training ─────────────────────────────────────────────────────
    parser.add_argument("--n_epochs",     type=int,   default=100)
    parser.add_argument("--epoch_size",   type=int,   default=128_000)
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--lr_model",     type=float, default=3e-5,
                        help="Actor LR (FIXED: was 1e-4)")
    parser.add_argument("--lr_critic",    type=float, default=3e-5,
                        help="Critic LR (FIXED: was 1e-4)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ── Evaluation ───────────────────────────────────────────────────
    parser.add_argument("--val_size",       type=int, default=10_000)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--decode_strategy", type=str, default="greedy")

    # ── Clustering ───────────────────────────────────────────────────
    parser.add_argument("--num_clusters", type=int,   default=0)

    # ── Sensitivity ──────────────────────────────────────────────────
    parser.add_argument("--perturbation_strength", type=float, default=0.05)
    parser.add_argument("--perturbation_freq",     type=str,   default="episode")

    # ── Logging / reproducibility ────────────────────────────────────
    parser.add_argument("--seed",              type=int, default=1234)
    parser.add_argument("--run_name",          type=str, default="qap_drl_run")
    parser.add_argument("--output_dir",        type=str, default="outputs")
    parser.add_argument("--log_dir",           type=str, default="logs")
    parser.add_argument("--checkpoint_epochs", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")

    # ── Resume / eval-only ───────────────────────────────────────────
    parser.add_argument("--load_path",  type=str, default=None)
    parser.add_argument("--eval_only",  action="store_true")

    opts = parser.parse_args(args)

    # Auto-set capacity
    if opts.capacity is None:
        opts.capacity = CAPACITY_MAP[opts.graph_size]

    return opts
