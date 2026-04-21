# KNOWLEDGE: QAP-DRL Architecture & Methodology

**Purpose:** Complete mathematical specification and PyTorch implementation reference for every QAP-DRL component.
**When to reference:** Any coding, implementation, or architecture question. This is the single source of truth.
**Last updated:** May 2026
**Key reference paper:** Giang et al. (2025) — "Vehicle Routing Problems via Quantum Graph Attention Network Deep Reinforcement Learning" (Q-GAT)
  → QAP-DRL is the classical lightweight alternative to Q-GAT's real PQC approach.
  → Q-GAT training config defines the benchmark: 100 epochs, batch=256, Adam lr=1e-4, val/test 10K instances, capacity (20,30),(50,40),(100,50).

---

## 0. Methodology Changes Applied (May 2026)

Two changes fix proximity-blindness in decoder scoring.
Changes 1+2 are implemented and canonical. Total new parameters: +5.

### Change 1 — Distance Proximity Penalty (§3.3.4 Scoring)

**Problem diagnosed:** The two existing scoring terms (attention + interference) are both
amplitude-space signals on S¹. Neither measures Euclidean distance from the vehicle's current
position to each candidate node.

**Formula change:**
```
Before:  Score(j) = q · ψ'ⱼ  +  λ · E_kNN(j)
After:   Score(j) = q · ψ'ⱼ  +  λ · E_kNN(j)  −  μ · dist(vₜ, vⱼ)
```
- `μ ∈ ℝ` = learnable scalar, init 0.5, jointly optimized by PPO
- File: `decoder/hybrid_scoring.py` — `self.mu_param = nn.Parameter(torch.tensor(0.5))`
- **+1 parameter**

### Change 2 — Spatial Context Grounding (§3.3.3 Context)

**Problem diagnosed:** At t=0, ψ'_curr = [0,0] so the query direction is determined only
by the capacity column of Wq — pointing arbitrarily, unrelated to node proximity.

**Formula change:**
```
Before:  ctx = [ψ'_curr(2), cap/C(1), t/N(1)]              ∈ ℝ⁴
After:   ctx = [ψ'_curr(2), cap/C(1), t/N(1), x_curr(1), y_curr(1)]  ∈ ℝ⁶
```
- `x_curr, y_curr` = actual Euclidean coordinates of current vehicle node ∈ [0,1]²
- At depot: x_curr = depot_x, y_curr = depot_y (NOT zero)
- Wq: ℝ^{2×4} → ℝ^{2×6}
- File: `decoder/context_query.py` — `self.Wq = nn.Linear(6, 2, bias=False)`
- `forward()` now returns `(query [B,2], current_coords [B,2])` — **always unpack both**
- **+4 parameters**

### Change 3 — Dynamic Proximity Feature (§3.3.1)
Not implemented. Encoder is STATIC: 5D features, computed once.

**Total across Changes 1+2: +5 parameters. Full model: ~391 → ~396.**

---

## 1. Problem Formulation (Thesis §3.1)

### CVRP as MDP

- **Graph:** Complete graph G = (V, E) where V = {0, 1, ..., N}. Node 0 = depot.
- **Vehicles:** K homogeneous vehicles, each with capacity C.
- **State:** Current node, visited set, remaining capacity per vehicle.
- **Action:** Select next unvisited customer (or return to depot to start new route).
- **Reward:** Negative travel cost: r_t = -c(node_t, node_{t+1}).
- **Objective:** Minimize total travel distance across all routes.
- **Termination:** All N customers visited and all vehicles returned to depot.

### Problem Sizes & Capacities (Thesis Table 3-4)

| Problem | N (customers) | C (capacity) | K (vehicles) |
|---------|--------------|-------------|-------------|
| CVRP-20 | 20 | 30 | dynamic |
| CVRP-50 | 50 | 40 | dynamic |
| CVRP-100 | 100 | 50 | dynamic |

K is not fixed — vehicles are dispatched as needed. A new route starts whenever the agent returns to depot.

### Data Generation Protocol

```python
coords_depot = torch.FloatTensor(1, 2).uniform_(0, 1)
coords_customers = torch.FloatTensor(N, 2).uniform_(0, 1)
demands = torch.randint(1, 10, (N,))
# Training: 128,000 instances/epoch, regenerated each epoch
# Validation: 10,000 fixed instances (VAL_EVAL_SIZE=10_000)
# OR-Tools ref: 1,000 instances (ORTOOLS_EVAL_SIZE=1_000)
```

---

## 2. Architecture Overview

```
Input: coords [B, N+1, 2], demands [B, N+1], capacity C
    │
    ▼
┌─────────────────────── ENCODER (STATIC — runs once) ──────────────────────────┐
│  ① Feature Construction    → [B, N+1, 5]                                     │
│     xᵢ = [d/C, dist_depot, x, y, angle/π]                                    │
│  ② Amplitude Projection    Linear(5→2) + L2 norm → [B, N+1, 2]               │
│     W ∈ ℝ^{2×5}                                                               │
│  ③ Rotation MLP            MLP(5→16→1) → θ_i → [B, N+1]                     │
│  ④ Rotation Matrix         R(θ_i) · ψ_i → ψ'_i → [B, N+1, 2]               │
│  NOTE: encoder called ONCE. psi_prime fixed for all decode steps.             │
└───────────────────────────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────── DECODER (autoregressive) ──────────────────────┐
│  For each step t = 0, 1, ..., N-1:                                     │
│  ⑤ Context Query   [ψ'_curr, cap/C, t/N, x_curr, y_curr] → q  [B,2]  │
│     ctx ∈ ℝ⁶, W_q ∈ ℝ^{2×6}  ← Change 2                             │
│  ⑥ Hybrid Scoring  q·ψ'_j + λ·E_kNN(j) − μ·dist(vₜ,vⱼ) [B, N+1]    │
│     −μ·dist term  ← Change 1                                          │
│  ⑦ Masking + Softmax → action probabilities → [B, N+1]                │
│  ⑧ Sample/Greedy → next node → [B]                                    │
└───────────────────────────────────────────────────────────────────────┘
    │ complete tour
    ▼
┌─────────────────────── PPO TRAINING ──────────────────────────────────┐
│  Actor:  π(a|s) from decoder softmax                                   │
│  Critic: MLP on mean-pooled ψ' → V(s) scalar                          │
│  Loss:   L_clip + c1·L_value + c2·L_entropy                           │
└───────────────────────────────────────────────────────────────────────┘
```

### Parameter Budget (Changes 1+2)

| Component | Specification | Parameters |
|-----------|--------------|------------|
| W (amplitude projection) | 2 × 5 | 10 |
| b (projection bias) | 2 × 1 | 2 |
| MLP (rotation) | 5→16→1 | ~113 |
| W_q (query projection) | **2 × 6**, no bias (+4 Change 2) | **12** |
| λ (interference balance) | scalar | 1 |
| μ (distance penalty) | **scalar** (+1 Change 1) | **1** |
| Critic MLP | 2→64→1 | ~257 |
| **Actor total** | | **~139** |
| **Full total** | | **~396** |

### Ablation Baseline (variant b)

| Component | Specification | Parameters |
|-----------|--------------|------------|
| BaselineEncoder | Linear(5→2) + ReLU — static features only | 12 |
| W_q (6), λ, μ, Critic | same as full model | 271 |
| **Baseline total** | | **~283** |

---

## 3. Encoder Components

### 3.1 Feature Construction (§3.X.3)

```
xᵢ = [dᵢ/C, dist(i,depot), xᵢ, yᵢ, atan2(Δy,Δx)/π]  ∈ ℝ⁵
      [0]       [1]          [2]  [3]      [4]
```

All 5 features are STATIC — computed once before the decoding loop.

`FeatureBuilder.forward(state)` — single argument, returns `[B, N+1, 5]`.

### 3.2 Amplitude Projection (§3.X.4)

```
ψ_i = Normalize(W_proj · xᵢ + b_proj),   W ∈ ℝ^{2×5},  α²+β²=1
```

### 3.3 Rotation MLP (§3.X.5)

```
θ_i = MLP(xᵢ)   MLP: ℝ^5 → ℝ^16 → ℝ^1, tanh
ψ'_i = R(θ_i) · ψ_i     (norm preserved)
```

### 3.4 FullEncoder interface

```python
# Static encode — called once before decoding loop:
psi_prime, features, knn_indices = encoder.forward(state)
# psi_prime: [B, N+1, 2], features: [B, N+1, 5], knn_indices: [B, N+1, k]
```

---

## 4. Decoder Components

### 4.1 Context Query Construction (§3.X.6) — Change 2

```
context_t = [ψ'_{curr}(2D),  cap_remaining/C,  t/N,  x_curr,  y_curr]  ∈ ℝ⁶
q_t       = W_q · context_t,    W_q ∈ ℝ^{2×6},  no bias
```

**ψ′_curr = zero vector if at depot. x_curr, y_curr = actual depot coordinates at depot.**

`context_query.forward()` returns `(query [B,2], current_coords [B,2])`. Always unpack both.

### 4.2 Hybrid Scoring Mechanism (§3.X.7) — Change 1

```
Score(j) = q_t · ψ'_j  +  λ · E_kNN(j)  −  μ · dist(vₜ, vⱼ)

λ learnable (init 0.1), μ learnable (init 0.5)
```

`HybridScoring.forward(query, psi_prime, knn_indices, mask, current_coords, all_coords)`

### 4.3 Decoder rollout

```python
actions, log_probs, tour_len = decoder.rollout(
    psi_prime, env_state, knn_indices, env, greedy=False
)
```

---

## 5. Critic Network

**UNCHANGED.** MLP: 2→64→1, ~257 params. `psi_prime DETACHED before critic head.`

---

## 6. PPO Training (§3.X.8)

### Hyperparameters (Phase 1b — v8/v9)

| Parameter | Value | Notes |
|-----------|-------|-------|
| K | 3 | |
| ε | 0.2 | |
| γ | 0.99 | |
| λ_GAE | 0.95 | |
| c1 | 0.5 | |
| c2 | 0.01 | 0.05 caused adv collapse |
| lr | 1e-4 | |
| eta_min | 1e-5 | v4 fix: was 1e-6 |
| batch_size | 512 | Phase 1b |
| kNN k | 10 | Phase 1 |
| λ_init | 0.1 | |
| μ_init | 0.5 | Change 1 |
| VAL_EVAL_SIZE | 10,000 | key paper standard |
| ORTOOLS_EVAL_SIZE | 1,000 | OR-Tools subset (speed) |

### evaluate_augmented() Rule
Coordinate augmentation + greedy decoding (evaluate.py v3).
8 isometric transforms of unit square × greedy decoding.

### OR-Tools Route Visualization
Post-training: `solve_one_with_routes()` runs OR-Tools on the SAME instance (`best_i`)
that produced `best_route.png`. Saves `ortools_route.png` with identical `plot_route_map()`
style for direct visual comparison of model vs optimal routes.

---

## 7. Canonical Math (Changes 1+2)

```
xᵢ    = [d_i/C, ‖i-depot‖, x_i, y_i, atan2/π]  ∈ ℝ⁵   (STATIC)
ψ_i    = normalize(W·xᵢ+b),  W ∈ ℝ^{2×5},  ‖ψ‖=1
θ_i    = MLP(xᵢ),  MLP: 5→16→1, tanh
ψ'_i   = R(θ_i)·ψ_i,  ‖ψ'‖=1
ctx_t  = [ψ'_curr, cap_t/C, t/N, x_curr, y_curr]  ∈ ℝ⁶            ← Change 2
q_t    = W_q·ctx_t,   W_q ∈ ℝ^{2×6}                               ← Change 2
Score(j) = q_t·ψ'_j + λ·Σ_{kNN}(ψ'_i·ψ'_j) − μ·‖coord_j−coord_curr‖  ← Change 1
L        = L_clip + c1·L_value + c2·L_entropy
```

---

## 8. Glossary (Changes 1+2)

| Thesis | Code | Shape |
|--------|------|---------|
| `xᵢ` | `features` | `[B,N+1,5]` |
| `ψ_i` | `psi` | `[B,N+1,2]` |
| `θ_i` | `theta` | `[B,N+1]` |
| `ψ'_i` | `psi_prime` | `[B,N+1,2]` (STATIC) |
| `ctx_t` | `ctx` | `[B,6]` ← was 4 |
| `q_t` | `query` | `[B,2]` |
| `Score(j)` | `scores` | `[B,N+1]` |
| `λ` | `self.lambda_param` | `nn.Parameter` |
| `μ` | `self.mu_param` | `nn.Parameter` ← Change 1 |
| `dist(vₜ,vⱼ)` | `dist_to_nodes` | `[B,N+1]` ← Change 1 |
| `x_curr,y_curr` | `current_coords` | `[B,2]` ← Change 2 |
| `kNN(j)` | `knn_indices` | `[B,N+1,k]` |
| `V(s_t)` | `value` | `[B]` |
| `A_t` | `advantages` | `[B,T]` |
