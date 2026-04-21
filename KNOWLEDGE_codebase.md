# KNOWLEDGE: Codebase Structure & Implementation Guide

**Purpose:** Define the exact file structure, naming conventions, and implementation order for the QAP-DRL codebase.
**When to reference:** Creating new files, understanding project layout, implementing components.
**Last updated:** May 2026

---

## 0. Methodology Changes Applied (May 2026)

All **two** changes are permanently part of the codebase. Total new params: +5.

### Change 1 — Distance Proximity Penalty (§3.3.4)
- File: `decoder/hybrid_scoring.py`
- Score formula: `Score(j) = q·ψ'ⱼ + λ·E_kNN(j) − μ·dist(vₜ, vⱼ)`
- `μ = nn.Parameter(torch.tensor(0.5))` in `HybridScoring`
- `forward()` gains two new args: `current_coords [B,2]`, `all_coords [B,N+1,2]`
- +1 parameter

### Change 2 — Spatial Context Grounding (§3.3.3)
- File: `decoder/context_query.py`
- Context: `ctx = [ψ'_curr(2), cap/C(1), t/N(1), x_curr(1), y_curr(1)] ∈ ℝ⁶`
- `Wq ∈ ℝ^{2×6}` (was 2×4)
- `forward()` returns `(query [B,2], current_coords [B,2])` — always unpack both
- +4 parameters

### Change 3 — Dynamic Proximity Feature (§3.3.1)
Not implemented. Encoder is STATIC: 5D features, computed once.

**Other files affected by Changes 1+2:**
- `decoder/qap_decoder.py` — passes `current_coords` and `state["coords"]` to scoring
- `models/qap_policy.py` — `context_dim=6`, `feature_dim=5`, `mu_init` arg, broadcast psi_prime
- `training/ppo_agent.py` — `update()` returns `mu_val`
- `train_n20.py` — logs `mu_val`, console column added, chart panel 8 shows λ+μ curves

---

## 1. Target Directory Structure

```
THESIS CODE QAP_VRP/
├── CLAUDE.md                        ← persistent reference (root)
├── cvrp-ppo/                        ← PRIMARY implementation folder
│   ├── run.py                       ← Main entry point (train/eval)
│   ├── options.py                   ← All hyperparameters & CLI args
│   ├── train_n20.py                 ← CVRP-20 training  [v9 Phase 1b + C1+C2]
│   ├── train_n50.py                 ← CVRP-50 training  [v8 Phase 1b]
│   ├── train_n100.py                ← CVRP-100 training
│   ├── train_n10.py                 ← CVRP-10 training
│   ├── train_ablation_n20.py        ← Ablation study: QAP-DRL vs Pure DRL
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── feature_constructor.py   ← Static 5D features
│   │   ├── amplitude_projection.py  ← input_dim=5, W 2×5
│   │   ├── rotation_mlp.py          ← input_dim=5, 5→16→1
│   │   ├── rotation.py              ← UNCHANGED
│   │   ├── qap_encoder.py           ← input_dim=5, static forward(state)
│   │   └── baseline_encoder.py      ← Ablation: plain MLP, no norm/rotation
│   ├── decoder/
│   │   ├── __init__.py
│   │   ├── context_query.py         ← UPDATED (Change 2): ctx ℝ⁴→ℝ⁶, Wq 2×4→2×6
│   │   │                               returns (query [B,2], current_coords [B,2])
│   │   ├── hybrid_scoring.py        ← UPDATED (Change 1): +μ·dist penalty, mu_param
│   │   └── qap_decoder.py           ← UPDATED: C1+C2 current_coords to scoring (no encoder arg)
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── cvrp_env.py              ← CVRP environment (reset, step, reward)
│   │   └── state.py                 ← StateCVRP dataclass
│   ├── models/
│   │   ├── __init__.py
│   │   └── qap_policy.py            ← UPDATED: context_dim=6, feature_dim=5, mu_init, broadcast psi_prime
│   ├── training/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py             ← UPDATED (v5): update() returns mu_val in diag dict
│   │   ├── rollout_buffer.py        ← Experience buffer for PPO
│   │   └── evaluate.py              ← UPDATED (v3): coord-aug+greedy (was stochastic+same)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── knn.py                   ← kNN precomputation (k=10 default for N=20)
│   │   ├── clustering.py            ← K-Means decomposition (§3.X.2)
│   │   ├── data_generator.py        ← CVRP instance generation (Kool et al. protocol)
│   │   ├── seed.py                  ← Seed management
│   │   ├── logger.py                ← TensorBoard + console logging
│   │   ├── checkpoint.py            ← Save/load model checkpoints
│   │   ├── metrics.py               ← Tour length, optimality gap, etc.
│   │   ├── ortools_refs.py          ← OR-Tools reference management
│   │   └── ortools_solver.py        ← UPDATED (v3): +solve_one_with_routes()
│   ├── configs/
│   │   └── default.yaml             ← YAML config (mirrors options.py)
│   └── outputs/
│       └── n20/                     ← CVRP-20 results
│           ├── train_log.jsonl      ← epoch logs (includes mu_val from v9)
│           ├── training_curves.png  ← 8-panel chart (λ+μ in panel 8, no twinx gridlines)
│           ├── best_model.pt
│           ├── best_route.png       ← model best route (greedy, best val instance)
│           ├── ortools_route.png    ← NEW: OR-Tools route on SAME instance as best_route
│           ├── cluster_map.png
│           └── Archive/             ← Previous run checkpoints
└── ref_code/                        ← READ-ONLY reference
```

---

## 2. Canonical Interfaces (post-Changes)

### context_query.ContextAndQuery.forward()
```python
def forward(self, state, psi_prime, step, n_customers):
    # Returns TUPLE — always unpack both values
    query, current_coords = context_query.forward(...)
    # query:          [B, 2]
    # current_coords: [B, 2]  — passed to HybridScoring
```

### hybrid_scoring.HybridScoring.forward()
```python
def forward(self, query, psi_prime, knn_indices, mask,
            current_coords,   # [B, 2]      NEW — Change 1
            all_coords):      # [B, N+1, 2] NEW — Change 1
    # Returns: log_probs [B, N+1]
```

### training/ppo_agent.PPOTrainer.update()
```python
# Returns dict with 16 fields (was 15):
{
    "policy_loss":   float,
    "value_loss":    float,
    "entropy":       float,
    "grad_norm":     float,
    "clip_fraction": float,
    "ratio_mean":    float,
    "adv_mean":      float,
    "adv_std":       float,
    "train_tour":    float,
    "greedy_tour":   float,
    "improvement":   float,
    "lambda_val":    float,   # λ interference weight
    "mu_val":        float,   # μ distance penalty  ← NEW Change 1
    "lr":            float,
}
```

### train_n20.py console columns
```
Ep | val_tour | vs ORT | actor_L | critic_L | entropy | grad | clip% | adv_std | λ | μ | feas% | t(s)
```
`μ` column shows current `mu_val` — healthy range 0.2–2.0.

### train_log.jsonl fields (v9 addition)
```json
{"step": 1, "val_tour": ..., "mu_val": 0.5, "lambda_val": 0.1, ...}
```

---

## 3. Implementation Order (current status)

**Phases 1–4: COMPLETE. Changes 1+2: COMPLETE.**

```
Phase 1b (active): python train_n20.py
                   python train_n50.py
Phase ablation:    python train_ablation_n20.py
Phase 2 (arch):    amplitude dim 2→4, rotation hidden 16→32  (after Phase 1b < 15% gap)
Phase 3 (longer):  400 epochs, CosineAnnealingWarmRestarts   (after Phase 2)
```

---

## 4. Parameter Budget (post-Changes)

| Component | File | Params |
|-----------|------|--------|
| W, b (amplitude proj) | encoder/amplitude_projection.py | 12 |
| MLP rotation | encoder/rotation_mlp.py | 113 |
| Wq (query proj, 2×6) | decoder/context_query.py | **12** (+4 Change 2) |
| λ (interference) | decoder/hybrid_scoring.py | 1 |
| μ (distance penalty) | decoder/hybrid_scoring.py | **1** (+1 Change 1) |
| Critic MLP (2→64→1) | models/qap_policy.py | 257 |
| **Total (QAP full)** | | **~396** |
| **Total (baseline)** | | **~283** |

---

## 5. File Modification Log

| File | Change | Version |
|------|--------|---------|
| `decoder/context_query.py` | ctx ℝ⁴→ℝ⁶, Wq 2×4→2×6, returns (query, coords) | Change 2 |
| `decoder/hybrid_scoring.py` | +mu_param, −μ·dist term in forward() | Change 1 |
| `decoder/qap_decoder.py` | passes current_coords+all_coords to hybrid | Change 1+2 |
| `models/qap_policy.py` | context_dim=6, mu_init arg, evaluate_actions updated | Change 1+2 |
| `training/ppo_agent.py` | update() returns mu_val in diagnostic dict | v5 |
| `train_n20.py` | mu_val logged, console column added, chart panel 8 updated | v9 |

**Unchanged:** `environment/`, `training/rollout_buffer.py`, `utils/knn.py`, `utils/data_generator.py`, `utils/seed.py`, `train_n100.py`

**Recent additions (this session):**

### `training/evaluate.py` — v3
```
FIX: coordinate augmentation + greedy decoding.
  8 geometric transforms (rot×4 + reflect×4) applied to instance coords.
  GREEDY decoding on each (deterministic → consistent v_t sequence).
  Tour length computed on ORIGINAL coords (distance-invariant).
  torch.minimum across 8 greedy runs → valid diversity.
```

### `utils/ortools_solver.py` — v3 (+solve_one_with_routes)
```
Added solve_one_with_routes(coords_np, demands_np, capacity, time_limit):
  Returns (tour_length, routes: list[list[int]]) — actual vehicle routes.
  Routes = list of customer node indices per vehicle (depot excluded).
  Same solver params as solve_one(). Used for route visualization.
```

### `train_n20.py` and `train_n50.py` — OR-Tools route map
```
Post-training: OR-Tools route plotted on SAME instance as best_route.png.
  Calls solve_one_with_routes() on instance best_i (identical coords/demands).
  Saves ortools_route.png using same plot_route_map() style.
  Prints per-instance gap: model tour vs OR-Tools tour.
  Import added: solve_one_with_routes, ORTOOLS_OK from ortools_solver.
```

### `train_n20.py` and `train_n50.py` — twinx gridline fix
```
All twinx() axes now get 3 lines after creation:
  ax.set_zorder(primary.get_zorder() - 1)   # push behind primary grid
  ax.patch.set_visible(False)                # hide twinx background
  ax.grid(False)                             # disable twinx gridlines
Fixes dense grey gridlines in dual-axis panels.
```

---

## 6. Key Invariants

1. `psi` and `psi_prime` always have L2 norm = 1.0 — QAP mode only
2. `knn_indices` never contain self-loops (diagonal=inf before topk)
3. Depot (index 0) never masked as infeasible
4. Feasibility mask applied BEFORE softmax (set to -1e9)
5. `context_query.forward()` returns a 2-tuple — always unpack
6. `psi_prime` DETACHED before critic head in ppo_agent.update()
7. Feature order: `[d/C, dist_depot, x, y, angle/π]` — 5D
8. Angle feature normalized by π → range [-1, 1]
9. At depot: ψ'_curr = [0, 0] but x_curr = depot_x, y_curr = depot_y (actual coords)
10. `mu_param` must be `nn.Parameter`, not a plain float
