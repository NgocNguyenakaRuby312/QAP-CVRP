# KNOWLEDGE: Codebase Structure & Implementation Guide

**Purpose:** Define the exact file structure, naming conventions, and implementation order for the QAP-DRL codebase.
**When to reference:** Creating new files, understanding project layout, implementing components.
**Last updated:** April 2026

---

## 1. Target Directory Structure

```
THESIS CODE QAP_VRP/
├── CLAUDE.md                        ← persistent reference (root)
├── cvrp-ppo/                        ← PRIMARY implementation folder
│   ├── run.py                       ← Main entry point (train/eval)
│   ├── options.py                   ← All hyperparameters & CLI args
│   ├── train_n20.py                 ← Self-contained CVRP-20 training  [v8 Phase 1b]
│   ├── train_n50.py                 ← Self-contained CVRP-50 training  [v8 Phase 1b]
│   ├── train_n100.py                ← Self-contained CVRP-100 training
│   ├── train_n10.py                 ← Self-contained CVRP-10 training
│   ├── train_ablation_n20.py        ← Ablation study: QAP-DRL vs Pure DRL [NEW]
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── feature_constructor.py   ← 5D feature vector: [d/C, dist, x, y, angle/π]
│   │   ├── amplitude_projection.py  ← Linear(5→2) + L2 normalize
│   │   ├── rotation_mlp.py          ← MLP(5→16→1, tanh) for θ_i
│   │   ├── rotation.py              ← 2D rotation matrix application
│   │   ├── qap_encoder.py           ← Combines all encoder components + FullEncoder
│   │   └── baseline_encoder.py      ← Ablation: plain MLP, no norm/rotation [NEW]
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
│   │                                   accepts encoder_type="qap"|"baseline" [UPDATED]
│   ├── training/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py             ← PPO training loop [v4: eta_min=1e-5]
│   │   ├── rollout_buffer.py        ← Experience buffer for PPO
│   │   └── evaluate.py              ← evaluate() + evaluate_augmented() [UPDATED]
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── knn.py                   ← kNN precomputation (k=10 default for N=20)
│   │   ├── clustering.py            ← K-Means decomposition (§3.X.2)
│   │   ├── data_generator.py        ← CVRP instance generation (Kool et al. protocol)
│   │   ├── seed.py                  ← Seed management
│   │   ├── logger.py                ← TensorBoard + console logging
│   │   ├── checkpoint.py            ← Save/load model checkpoints
│   │   ├── metrics.py               ← Tour length, optimality gap, etc.
│   │   ├── ortools_refs.py          ← OR-Tools reference management [UPDATED: rich banner]
│   │   └── ortools_solver.py        ← OR-Tools GLS solver [UPDATED: percentiles + timing]
│   ├── configs/
│   │   └── default.yaml             ← YAML config (mirrors options.py)
│   ├── datasets/
│   │   ├── val_n20.pkl              ← 2.40 MB, 500+ instances
│   │   ├── val_n50.pkl              ← 5.84 MB
│   │   ├── val_n100.pkl             ← 11.56 MB
│   │   └── ortools_refs.json        ← Cached OR-Tools reference statistics
│   ├── outputs/
│   │   ├── n20/                     ← CVRP-20 training outputs
│   │   │   ├── train_log.jsonl      ← Per-epoch log (all metrics)
│   │   │   ├── best_model.pt        ← Best checkpoint
│   │   │   ├── training_curves.png  ← 8-panel chart (4×2)
│   │   │   ├── best_route.png
│   │   │   ├── cluster_map.png
│   │   │   ├── epochs/              ← Per-epoch checkpoints (epoch_001.pt ...)
│   │   │   └── Archive/             ← Previous runs auto-archived
│   │   ├── n50/                     ← CVRP-50 training outputs (same structure)
│   │   └── ablation_n20/            ← Ablation study outputs [NEW]
│   │       ├── qap/                 ← Full QAP-DRL results
│   │       ├── baseline/            ← Pure DRL baseline results
│   │       ├── ablation_comparison.png
│   │       └── ablation_results.json
│   └── tests/
│       ├── test_encoder.py
│       ├── test_decoder.py
│       ├── test_env.py
│       ├── test_shapes.py
│       └── test_smoke.py
└── ref_code/                        ← VRP-DACT reference (READ-ONLY)
```

---

## 2. Key File Changelogs

### `train_n20.py` — v8 Phase 1b (current)
```
v4 — ENTROPY_COEF 0.01 → 0.02
v5 — 6-panel chart
v6 — eta_min 1e-6 → 1e-5 (ppo_agent.py)
v7 — 8-panel chart (4×2)
v8 — Phase 1:
     ENTROPY_COEF 0.02 → 0.05
     EPOCH_SIZE 51,200 → 128,000
     KNN_K 5 → 10
     evaluate() → evaluate_augmented(×8)
v8 Phase 1b (current):
     ENTROPY_COEF 0.05 → 0.01  (0.05 caused advantage signal collapse)
     BATCH_SIZE 256 → 512      (wider adv distribution, stronger PPO signal)
     BATCHES_PER_EPOCH = 250   (auto: 128,000 ÷ 512)
     matplotlib.ticker import  (chart rendering fix)
     Explicit ylim on twinx panels (entropy, clip/lambda chart bugs fixed)
```

### `training/evaluate.py` — v2
```
Added evaluate_augmented(model, instances, device, n_samples=8):
  - n_samples stochastic rollouts per instance
  - torch.minimum element-wise across all samples
  - Returns mean of per-instance best tours
  - Zero retraining cost
```

### `utils/ortools_solver.py` — v2
```
solve_one() now returns (tour_length, solve_time) tuple
compute_and_save_ref() now stores:
  - p10, p25, p50, p75, p90 (percentile distribution)
  - mean_solve_time, max_solve_time, n_time_limited
```

### `utils/ortools_refs.py` — v2
```
Added _print_banner() — called on every ensure_ortools_ref() call
ensure_ortools_ref() gains output_dir parameter (for current best model gap)
Banner prints: mean, std, CV%, ±2σ range, percentiles, timing, 5% target, current best
```

### `models/qap_policy.py` — updated
```
Added encoder_type parameter: "qap" (default) | "baseline" (ablation)
When "baseline": uses FullBaselineEncoder instead of FullEncoder
Import of FullBaselineEncoder added
```

### `encoder/baseline_encoder.py` — NEW
```
BaselineEncoder: Linear(5→2) + ReLU, 12 params, no L2 norm, no rotation
FullBaselineEncoder: drop-in for FullEncoder, identical interface
Used by QAPPolicy(encoder_type="baseline") for ablation study
```

### `train_ablation_n20.py` — NEW
```
Runs QAP-DRL and Pure DRL baseline back-to-back under identical conditions
Same seed, hyperparams, data, validation for both
Produces: comparison chart, per-epoch logs, verdict table, ablation_results.json
```

---

## 3. Implementation Order (original phases complete — current focus)

### Phase 1–4: COMPLETE
All core components implemented and working.

### Phase 5: Active Work

| Task | Status | File |
|------|--------|------|
| Train CVRP-20 Phase 1b | Next run | `train_n20.py` |
| Train CVRP-50 Phase 1b | After n20 | `train_n50.py` |
| Ablation study (variant b) | Ready to run | `train_ablation_n20.py` |
| Phase 2 (amplitude dim 2→4) | After Phase 1b confirmed | TBD |
| Phase 3 (400 epochs + warm restarts) | After Phase 2 | TBD |

### Gap Reduction Roadmap

| Phase | Changes | Expected gap |
|-------|---------|-------------|
| Phase 1 run | ENTROPY=0.05, BATCH=256, kNN=10, aug×8 | ~17% (achieved) |
| Phase 1b (current) | ENTROPY=0.01, BATCH=512 | target ~12-15% |
| Phase 2 (arch) | amplitude dim 2→4, rotation hidden 16→32 | target ~7-10% |
| Phase 3 (longer) | 400 epochs, CosineWarmRestarts | target <5% |

---

## 4. File-by-File Mapping: DACT → QAP-DRL

| DACT File (ref_code/) | QAP-DRL File (cvrp-ppo/) | What Changes |
|------------------------|--------------------------|--------------|
| `nets/graph_layers.py` | `encoder/qap_encoder.py` | **Replace entirely.** DAC-Att transformer → amplitude projection + rotation |
| `nets/actor_network.py` | `decoder/qap_decoder.py` | **Replace entirely.** Improvement decoder → constructive autoregressive |
| `nets/critic_network.py` | `models/qap_policy.py` (critic part) | **Simplify.** Mean-pool ψ' → MLP(2→64→1) → scalar |
| `agent/ppo.py` | `training/ppo_agent.py` | **Restructure.** n-step improvement PPO → episode-level constructive PPO with GAE |
| `problems/vrp/state_vrp.py` | `environment/state.py` | **Rewrite.** Improvement state → constructive MDP state |
| `problems/vrp/vrp.py` | `environment/cvrp_env.py` | **Rewrite.** Solution refinement env → sequential selection env |

---

## 5. Key Implementation Rules

### Tensor Conventions
- **Batch-first:** All tensors shaped `[batch_size, ...]`
- **Depot at index 0:** Node indexing: 0 = depot, 1..N = customers
- **Shape comments:** Every tensor operation that changes shape MUST have a shape comment
- **Device:** Every tensor must be on the same `device`

### Critical Invariants
1. `psi` and `psi_prime` always have L2 norm = 1.0 per vector (atol=1e-5) — QAP mode only
2. `knn_indices` never contain self-loops (diagonal set to inf before topk)
3. Depot (index 0) is never masked as infeasible
4. Feasibility mask applied BEFORE softmax (set to -1e9)
5. Episode terminates when ALL N customers are visited
6. Demands are integers in [1, 9]; depot demand = 0
7. Rotation preserves unit norm (no re-normalization needed)
8. Angle feature is normalized by π → range [-1, 1]
9. `psi_prime` DETACHED before critic head in ppo_agent.update()
10. Run train scripts from inside `cvrp-ppo/` directory (path resolution)

### What NOT to Implement
- No curriculum learning (DACT-specific)
- No 2-opt operators or solution improvement
- No dummy depot nodes (DACT trick)
- No cyclic positional encoding (DACT-specific)
- No dual-aspect attention (DACT-specific)
- No PennyLane/Qiskit/quantum libraries

---

## 6. Training Configuration (Current — Phase 1b)

```python
# train_n20.py settings
GRAPH_SIZE        = 20
CAPACITY          = 30
BATCH_SIZE        = 512       # Phase 1b: was 256
N_EPOCHS          = 200
EPOCH_SIZE        = 128_000   # thesis spec
LR                = 1e-4
ENTROPY_COEF      = 0.01      # Phase 1b: was 0.05 (caused adv collapse)
VALUE_COEF        = 0.5
KNN_K             = 10        # Phase 1: was 5
AUG_SAMPLES       = 8         # inference augmentation
BATCHES_PER_EPOCH = 250       # 128,000 ÷ 512
TOTAL_OPT_STEPS   = 1_200_000 # 200 × 250 × 3 × 8
```

---

## 7. run.py Entry Point Flow

```python
# 1. Parse opts (options.py)
# 2. Set seeds (utils/seed.py)
# 3. Detect device — ONCE here, pass everywhere
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 4. Load OR-Tools reference (ensure_ortools_ref) — prints banner before training
# 5. Create data generator (utils/data_generator.py) — pass device
# 6. Create environment (environment/cvrp_env.py)
# 7. Create QAPPolicy (encoder_type="qap") → .to(device)
# 8. Create PPO agent (training/ppo_agent.py) — pass device
# 9. Load checkpoint if resuming (auto-resume from epochs/*.pt)
# 10. Training loop:
#    for epoch in range(n_epochs):
#        torch.cuda.empty_cache()
#        generate training data (fresh each epoch, on device)
#        collect rollouts (forward pass through env)
#        PPO update (K=3 inner epochs, GAE advantages, normalized)
#        evaluate_augmented() on validation set (n_samples=8)
#        log all 15+ metrics to train_log.jsonl
#        save epoch checkpoint
#        redraw 8-panel chart (training_curves.png)
# 11. Save best_model.pt, best_route.png, cluster_map.png
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
| CVRP-20 | 512 | ~1.0 GB | ✓ Safe (Phase 1b) |
| CVRP-50 | 512 | ~2.0 GB | ✓ Safe |
| CVRP-100 | 256 | ~2.0 GB | ✓ Safe default |
| CVRP-100 | 512 | ~4.0 GB | ⚠ Test first, may OOM |
