# KNOWLEDGE: Codebase Structure & Implementation Guide

**Purpose:** Define the exact file structure, naming conventions, and implementation order for the QAP-DRL codebase.
**When to reference:** Creating new files, understanding project layout, implementing components.
**Last updated:** March 2026

---

## 1. Target Directory Structure

```
THESIS CODE QAP_VRP/
├── CLAUDE.md                        ← persistent reference (root)
├── cvrp-ppo/                        ← PRIMARY implementation folder
│   ├── run.py                       ← Main entry point (train/eval)
│   ├── options.py                   ← All hyperparameters & CLI args
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── feature_constructor.py   ← 5D feature vector: [d/C, dist, x, y, angle/π]
│   │   ├── amplitude_projection.py  ← Linear(5→2) + L2 normalize
│   │   ├── rotation_mlp.py          ← MLP(5→16→1, tanh) for θ_i
│   │   ├── rotation.py              ← 2D rotation matrix application
│   │   └── qap_encoder.py           ← Combines all encoder components
│   ├── decoder/
│   │   ├── __init__.py
│   │   ├── context_query.py         ← ContextQueryLayer (4→2, no bias)
│   │   ├── hybrid_scoring.py        ← HybridScoringLayer (context + kNN interference)
│   │   └── qap_decoder.py           ← Autoregressive decoding loop
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── cvrp_env.py              ← CVRP environment (reset, step, reward)
│   │   └── state.py                 ← StateCVRP dataclass
│   ├── models/
│   │   ├── __init__.py
│   │   └── qap_policy.py            ← QAPPolicy (encoder + decoder + critic)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py             ← PPO training loop (GAE, clipped objective)
│   │   ├── rollout_buffer.py        ← Experience buffer for PPO
│   │   └── evaluate.py              ← Greedy evaluation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── knn.py                   ← kNN precomputation (spatial coords, no self-loops)
│   │   ├── clustering.py            ← K-Means decomposition (§3.X.2)
│   │   ├── data_generator.py        ← CVRP instance generation (Kool et al. protocol)
│   │   ├── seed.py                  ← Seed management
│   │   ├── logger.py                ← TensorBoard + console logging
│   │   ├── checkpoint.py            ← Save/load model checkpoints
│   │   └── metrics.py               ← Tour length, optimality gap, etc.
│   ├── configs/
│   │   └── default.yaml             ← YAML config (mirrors options.py)
│   ├── datasets/                    ← Generated validation/test data (.pkl)
│   ├── outputs/                     ← Checkpoints, logs, results
│   └── tests/
│       ├── test_encoder.py
│       ├── test_decoder.py
│       ├── test_env.py
│       ├── test_shapes.py
│       └── test_smoke.py            ← End-to-end CVRP-20 smoke test
└── ref_code/                        ← VRP-DACT reference (READ-ONLY)
```

---

## 2. Implementation Order

Follow this sequence. Each step depends on the previous.

### Phase 1: Foundation (Days 1-3)
1. `utils/seed.py` — seed management
2. `utils/data_generator.py` — CVRP instance generation
3. `environment/state.py` — StateCVRP dataclass
4. `environment/cvrp_env.py` — environment with reset/step/reward
5. `tests/test_env.py` — verify env produces valid episodes

### Phase 2: Encoder (Days 3-5)
6. `encoder/feature_constructor.py` — 5D features: `[d/C, dist, x, y, angle/π]`
7. `encoder/amplitude_projection.py` — Linear(5→2) + L2 norm
8. `encoder/rotation_mlp.py` — MLP(5→16→1, tanh) for θ
9. `encoder/rotation.py` — 2D rotation matrix
10. `encoder/qap_encoder.py` — full encoder pipeline
11. `tests/test_encoder.py` — verify shapes, unit norms, gradient flow

### Phase 3: Decoder (Days 5-7)
12. `utils/knn.py` — kNN precomputation (spatial coords, exclude self-loops)
13. `decoder/context_query.py` — context [ψ'_curr, cap/C, t/N] → query
14. `decoder/hybrid_scoring.py` — context attention + kNN interference + masking
15. `decoder/qap_decoder.py` — autoregressive loop with action selection
16. `tests/test_decoder.py` — verify decoding produces valid feasible tours

### Phase 4: Policy & Training (Days 7-10)
17. `models/qap_policy.py` — actor (encoder+decoder) + critic (mean-pool → MLP)
18. `training/rollout_buffer.py` — PPO buffer (states, actions, log_probs, rewards, values)
19. `training/ppo_agent.py` — PPO loop (GAE advantages, clipped loss, K=3 epochs)
20. `training/evaluate.py` — greedy evaluation on validation set
21. `options.py` — all hyperparameters
22. `run.py` — entry point
23. `tests/test_smoke.py` — end-to-end CVRP-20 training (5 epochs, verify reward improves)

### Phase 5: Extensions (Days 10+)
24. `utils/clustering.py` — K-Means for large instances (§3.X.2)

---

## 3. File-by-File Mapping: DACT → QAP-DRL

| DACT File (ref_code/) | QAP-DRL File (cvrp-ppo/) | What Changes |
|------------------------|--------------------------|--------------|
| `nets/graph_layers.py` | `encoder/qap_encoder.py` | **Replace entirely.** DAC-Att transformer → amplitude projection + rotation |
| `nets/actor_network.py` | `decoder/qap_decoder.py` | **Replace entirely.** Improvement decoder → constructive autoregressive |
| `nets/critic_network.py` | `models/qap_policy.py` (critic part) | **Simplify.** Mean-pool ψ' → MLP(2→64→1) → scalar |
| `agent/ppo.py` | `training/ppo_agent.py` | **Restructure.** n-step improvement PPO → episode-level constructive PPO with GAE |
| `problems/vrp/state_vrp.py` | `environment/state.py` | **Rewrite.** Improvement state → constructive MDP state |
| `problems/vrp/vrp.py` | `environment/cvrp_env.py` | **Rewrite.** Solution refinement env → sequential selection env |
| `options.py` | `options.py` | **Prune.** Remove DACT-specific args, add QAP-DRL args |
| `run.py` | `run.py` | **Simplify.** Remove CL, augmentation logic |
| `utils/functions.py` | `utils/knn.py` + `utils/clustering.py` | **Split.** Keep kNN + add clustering |

---

## 4. Key Implementation Rules

### Tensor Conventions
- **Batch-first:** All tensors shaped `[batch_size, ...]`
- **Depot at index 0:** Node indexing: 0 = depot, 1..N = customers
- **Shape comments:** Every tensor operation that changes shape MUST have a shape comment
- **Device:** Every tensor must be on the same `device` — never mix CPU and CUDA

### Feature Order (Thesis §3.X.3)
```
x_i = [d_i/C, dist(i,depot), x_i, y_i, α_i/π]
         [0]       [1]        [2]  [3]    [4]
```

### Critical Invariants
1. `psi` and `psi_prime` always have L2 norm = 1.0 per vector (atol=1e-5)
2. `knn_indices` never contain self-loops (diagonal set to inf before topk)
3. Depot (index 0) is never masked as infeasible
4. Feasibility mask applied BEFORE softmax (set to -1e9)
5. Episode terminates when ALL N customers are visited
6. Demands are integers in [1, 9]; depot demand = 0
7. Rotation preserves unit norm (no re-normalization needed)
8. Angle feature is normalized by π → range [-1, 1]
9. All tensors and models must be on the correct `device`

### What NOT to Implement
- No curriculum learning (DACT-specific)
- No 8× data augmentation at inference (DACT-specific)
- No 2-opt operators or solution improvement
- No dummy depot nodes (DACT trick)
- No cyclic positional encoding (DACT-specific)
- No dual-aspect attention (DACT-specific)
- No PennyLane/Qiskit/quantum libraries

### Code Style
- Python 3.10+ type hints
- Google-style docstrings with shape annotations
- `import torch.nn.functional as F`
- `from typing import Optional, Tuple, Dict`
- Constants in UPPER_CASE
- Module-level `__all__` exports
- Thesis equation references in comments

---

## 5. options.py — Complete Argument Set

```python
# Problem
--problem         'vrp'
--graph_size      [20, 50, 100]
--capacity        [30, 40, 50]     # auto-matched to graph_size

# QAP-DRL architecture
--embedding_dim   2                # amplitude space dimension (fixed)
--hidden_dim      16               # rotation MLP hidden size (thesis: 16)
--knn_k           5                # interference neighbors
--lambda_init     0.1              # learnable lambda initial value

# Clustering (§3.X.2, for scalability)
--num_clusters    0                # 0 = disabled

# PPO
--K_epochs        3
--eps_clip        0.2
--gamma           0.99
--gae_lambda      0.95
--c1              0.5
--c2              0.01

# Training schedule
--n_epochs        100
--epoch_size      128000
--batch_size      256              # RTX 3050 default (raise to 512 only if no OOM)
--lr_model        1e-4
--lr_critic       1e-4
--max_grad_norm   1.0

# Evaluation
--val_size        10000
--eval_batch_size 256
--decode_strategy 'greedy'

# Sensitivity analysis
--perturbation_strength  0.05
--perturbation_freq      'episode'

# Logging
--run_name        'qap_drl_run'
--output_dir      'outputs'
--log_dir         'logs'
--checkpoint_epochs 1
--seed            1234
```

---

## 6. Environment State Design

```python
@dataclass
class StateCVRP:
    """Constructive MDP state for CVRP."""
    coords: torch.Tensor          # [B, N+1, 2]
    demands: torch.Tensor         # [B, N+1]
    capacity: float               # scalar C
    visited: torch.Tensor         # [B, N+1] boolean
    current_node: torch.Tensor    # [B] int — current position
    remaining_cap: torch.Tensor   # [B] float — remaining vehicle capacity
    step: int                     # current decoding step (0 to N-1)
    tour: list                    # [B, step] — sequence of visited nodes
    total_distance: torch.Tensor  # [B] float — accumulated distance
    all_done: bool                # True when all customers visited
```

### Episode Flow
```
reset() → initial state (at depot, nothing visited, full capacity)
    ↓
for step in range(max_steps):
    action = policy(state)        # select next customer (or depot)
    state = env.step(action)      # update visited, capacity, distance
    if action == depot:           # vehicle returns to depot
        remaining_cap = C         # reset capacity for new route
    if all customers visited:
        break
done → return to depot, compute total distance
reward = -total_distance
```

---

## 7. run.py Entry Point Flow

```python
# 1. Parse opts (options.py)
# 2. Set seeds (utils/seed.py)
# 3. Detect device — ONCE here, pass everywhere
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
# 4. Create data generator (utils/data_generator.py) — pass device
# 5. Create environment (environment/cvrp_env.py)
# 6. Create QAPPolicy (encoder + decoder + critic) → .to(device)
# 7. Create PPO agent (training/ppo_agent.py) — pass device
# 8. Load checkpoint if --load_path
# 9. If --eval_only: evaluate and exit
# 10. Training loop:
#    for epoch in range(n_epochs):
#        torch.cuda.empty_cache()          ← clear VRAM each epoch
#        generate training data (fresh each epoch, on device)
#        collect rollouts (forward pass through env)
#        PPO update (K_epochs inner loop, GAE advantages)
#        evaluate on validation set (greedy, torch.no_grad())
#        log: reward, tour_length, feasibility_rate, entropy, loss
#        log VRAM usage if CUDA
#        save checkpoint
# 11. Final evaluation on test set
```

---

## 8. GPU / Device Rules Per File

| File | Device Rule |
|------|------------|
| `run.py` | Detect device here ONLY. Pass to all other modules. |
| `utils/data_generator.py` | Accept `device` argument. All tensors `.to(device)`. |
| `utils/knn.py` | `coords` already on device. Return `knn_indices` on same device. |
| `utils/clustering.py` | CPU-only (sklearn). Move tensors back to device after cluster. |
| `encoder/*.py` | Modules inherit device from `.to(device)` call in run.py. |
| `decoder/*.py` | Same — device follows model. |
| `models/qap_policy.py` | `model.to(device)` called in run.py, not inside the class. |
| `training/rollout_buffer.py` | Store tensors on device. Accept `device` argument. |
| `training/ppo_agent.py` | `torch.cuda.empty_cache()` at start of each epoch. |
| `training/evaluate.py` | Always inside `torch.no_grad()`. Tensors on device. |

### VRAM Budget (RTX 3050, 4GB)

| Problem | batch_size | Est. VRAM | Recommendation |
|---------|-----------|-----------|----------------|
| CVRP-20 | 256 | ~0.5 GB | ✓ Safe default |
| CVRP-50 | 256 | ~1.0 GB | ✓ Safe default |
| CVRP-100 | 256 | ~2.0 GB | ✓ Safe default |
| CVRP-100 | 512 | ~4.0 GB | ⚠ Test first, may OOM |

If OOM: reduce to `batch_size=128`, add `torch.cuda.empty_cache()` more frequently.
