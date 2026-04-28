# KNOWLEDGE: QAP-DRL Architecture & Methodology

**Purpose:** Complete mathematical specification and PyTorch implementation reference for every QAP-DRL component.
**When to reference:** Any coding, implementation, or architecture question. This is the single source of truth.
**Last updated:** April 2026 (Phase 2: 4D amplitudes)
**Key reference paper:** Giang et al. (2025) — Q-GAT
**Parameter budget:** ~609 (under 618 ceiling)

---

## 0. Methodology Changes Applied

### Change 1 — Distance Proximity Penalty (§3.3.4 Scoring)
```
Score(j) = q · ψ'ⱼ  +  λ · E_kNN(j)  −  μ · dist(vₜ, vⱼ)
```
- `μ` = learnable scalar, init 0.5, clamp [0, 20]
- File: `decoder/hybrid_scoring.py`

### Change 2 — Spatial Context Grounding (§3.3.3 Context)
```
ctx = [ψ'_curr(4), cap/C(1), t/N(1), x_curr(1), y_curr(1)]  ∈ ℝ⁸
```
- Wq ∈ ℝ^{4×8}, returns `(query [B,4], current_coords [B,2])`
- File: `decoder/context_query.py`

### Change 3 — Dynamic Proximity Feature (§3.3.1)
Not implemented. Encoder is STATIC: 5D features, computed once.

### Phase 2 — 4D Amplitudes (S³ Hypersphere)
- ψ ∈ ℝ² → ψ ∈ ℝ⁴ (unit circle → unit hypersphere S³)
- Rotation: single θ → 6 Givens angles for SO(4)
- MLP: 5→16→1 → 5→32→6 (hidden_dim=32)
- Context: ℝ⁶ → ℝ⁸ (D+4 where D=4)
- Critic: 2→64→1 → 4→64→1
- μ clamp [0, 20], λ clamp [-2, 3]
- CosineAnnealingWarmRestarts(T_0=50, T_mult=2)
- No weight_decay on μ

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

### Problem Sizes & Capacities

| Problem | N | C | K |
|---------|---|---|---|
| CVRP-20 | 20 | 30 | dynamic |
| CVRP-50 | 50 | 40 | dynamic |
| CVRP-100 | 100 | 50 | dynamic |
| CVRP-200 | 200 | 50 | dynamic |

---

## 2. Architecture Overview

```
Input: coords [B, N+1, 2], demands [B, N+1], capacity C
    │
    ▼
┌─────────────────────── ENCODER (STATIC — runs once) ──────────────────────────┐
│  ① Feature Construction    → [B, N+1, 5]                                     │
│     xᵢ = [d/C, dist_depot, x, y, angle/π]                                    │
│  ② Amplitude Projection    Linear(5→4) + L2 norm → [B, N+1, 4]  (S³)        │
│     W ∈ ℝ^{4×5}, b ∈ ℝ^4                                                     │
│  ③ Rotation MLP            MLP(5→32→6) → 6 Givens angles → [B, N+1, 6]      │
│  ④ SO(4) Rotation          G₆·G₅·G₄·G₃·G₂·G₁ · ψ → ψ' → [B, N+1, 4]      │
│     Planes: (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)                                   │
│  NOTE: encoder called ONCE. psi_prime fixed for all decode steps.             │
└───────────────────────────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────── DECODER (autoregressive) ──────────────────────┐
│  For each step t = 0, 1, ..., N-1:                                     │
│  ⑤ Context Query   [ψ'_curr(4), cap/C, t/N, x_curr, y_curr] → q [B,4]│
│     ctx ∈ ℝ⁸, W_q ∈ ℝ^{4×8}                                          │
│  ⑥ Hybrid Scoring  q·ψ'_j + λ·E_kNN(j) − μ·dist(vₜ,vⱼ) [B, N+1]    │
│     μ clamp [0,20], λ clamp [-2,3]                                     │
│  ⑦ Masking + Softmax → action probabilities → [B, N+1]                │
│  ⑧ Sample/Greedy → next node → [B]                                    │
└───────────────────────────────────────────────────────────────────────┘
    │ complete tour
    ▼
┌─────────────────────── PPO TRAINING ──────────────────────────────────┐
│  Actor:  π(a|s) from decoder softmax                                   │
│  Critic: MLP on mean-pooled ψ' → V(s) scalar  (4→64→1)               │
│  Loss:   L_clip + c1·L_value + c2·L_entropy                           │
│  Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2)             │
└───────────────────────────────────────────────────────────────────────┘
```

### Parameter Budget (Phase 2: 4D + hidden=32)

| Component | Specification | Parameters |
|-----------|--------------|------------|
| W (amplitude projection) | 4 × 5 | 20 |
| b (projection bias) | 4 × 1 | 4 |
| MLP (rotation) | 5→32→6 | ~230 |
| W_q (query projection) | 4 × 8, no bias | 32 |
| λ (interference balance) | scalar | 1 |
| μ (distance penalty) | scalar | 1 |
| Critic MLP | 4→64→1 | ~321 |
| **Actor total** | | **~288** |
| **Full total** | | **~609** |

### Ablation Baseline

| Component | Specification | Parameters |
|-----------|--------------|------------|
| BaselineEncoder | Linear(5→4) + ReLU — static features only | 24 |
| W_q (8), λ, μ, Critic | same as full model | 355 |
| **Baseline total** | | **~379** |

---

## 3. Encoder Components

### 3.1 Feature Construction (§3.X.3)
```
xᵢ = [dᵢ/C, dist(i,depot), xᵢ, yᵢ, atan2(Δy,Δx)/π]  ∈ ℝ⁵
      [0]       [1]          [2]  [3]      [4]
```
All 5 features are STATIC — computed once before the decoding loop.

### 3.2 Amplitude Projection (§3.X.4)
```
ψ_i = Normalize(W · xᵢ + b),   W ∈ ℝ^{4×5}, b ∈ ℝ^4,  ‖ψ‖=1 on S³
```

### 3.3 Rotation (§3.X.5)
```
Θ_i = MLP(xᵢ)   MLP: ℝ^5 → ℝ^32 → ℝ^6, tanh  (6 Givens angles for SO(4))
ψ'_i = R(Θ_i) · ψ_i     R = G₆·G₅·G₄·G₃·G₂·G₁   (norm preserved)
Planes: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
```

### 3.4 FullEncoder interface
```python
psi_prime, features, knn_indices = encoder.forward(state)
# psi_prime: [B, N+1, 4], features: [B, N+1, 5], knn_indices: [B, N+1, k]
```

---

## 4. Decoder Components

### 4.1 Context Query Construction (§3.X.6)
```
context_t = [ψ'_{curr}(4D),  cap_remaining/C,  t/N,  x_curr,  y_curr]  ∈ ℝ⁸
q_t       = W_q · context_t,    W_q ∈ ℝ^{4×8},  no bias
```
ψ′_curr = zero vector if at depot. x_curr, y_curr = actual depot coordinates at depot.
Returns `(query [B,4], current_coords [B,2])`. Always unpack both.

### 4.2 Hybrid Scoring Mechanism (§3.X.7)
```
Score(j) = q_t · ψ'_j  +  λ · E_kNN(j)  −  μ · dist(vₜ, vⱼ)
λ learnable (init 0.1), clamp [-2, 3]
μ learnable (init 0.5), clamp [0, 20]
```

### 4.3 Decoder rollout
```python
actions, log_probs, tour_len = decoder.rollout(
    psi_prime, env_state, knn_indices, env, greedy=False
)
```

---

## 5. Critic Network

MLP: 4→64→1, ~321 params. `psi_prime DETACHED before critic head.`

---

## 6. PPO Training (§3.X.8)

### Hyperparameters (Phase 2+)

| Parameter | Value | Notes |
|-----------|-------|-------|
| K (PPO epochs) | 3 | |
| ε (clip) | 0.2 | |
| γ | 0.99 | |
| λ_GAE | 0.95 | |
| c1 (value) | 0.5 | |
| c2 (entropy) | 0.03 | was 0.01 |
| lr | 1e-4 | |
| Scheduler | WarmRestarts(T_0=50, T_mult=2) | restarts at ep 50, 150 |
| eta_min | 1e-5 | |
| batch_size | 512 | |
| kNN k | 10 (N=20), 15 (N=50), 20 (N=100), 30 (N=200) | |
| AMP_DIM | 4 | S³ hypersphere |
| HIDDEN_DIM | 32 | rotation MLP |
| μ clamp | [0, 20] | |
| λ clamp | [-2, 3] | |
| μ weight_decay | 0 (none) | |

---

## 7. Canonical Math (Phase 2: 4D)

```
xᵢ    = [d_i/C, ‖i-depot‖, x_i, y_i, atan2/π]  ∈ ℝ⁵   (STATIC)
ψ_i    = normalize(W·xᵢ+b),  W ∈ ℝ^{4×5},  ‖ψ‖=1 on S³
Θ_i    = MLP(xᵢ),  MLP: 5→32→6, tanh  (6 Givens angles)
ψ'_i   = R(Θ_i)·ψ_i,  ‖ψ'‖=1   (SO(4) rotation)
ctx_t  = [ψ'_curr(4), cap_t/C, t/N, x_curr, y_curr]  ∈ ℝ⁸
q_t    = W_q·ctx_t,   W_q ∈ ℝ^{4×8}
Score(j) = q_t·ψ'_j + λ·Σ_{kNN}(ψ'_i·ψ'_j) − μ·‖coord_j−coord_curr‖
L        = L_clip + c1·L_value + c2·L_entropy
```

---

## 8. Glossary (Phase 2: 4D)

| Thesis | Code | Shape |
|--------|------|-------|
| `xᵢ` | `features` | `[B,N+1,5]` |
| `ψ_i` | `psi` | `[B,N+1,4]` on S³ |
| `Θ_i` | `theta` | `[B,N+1,6]` 6 Givens angles |
| `ψ'_i` | `psi_prime` | `[B,N+1,4]` (STATIC) |
| `ctx_t` | `ctx` | `[B,8]` (D+4=8) |
| `q_t` | `query` | `[B,4]` |
| `Score(j)` | `scores` | `[B,N+1]` |
| `λ` | `self.lambda_param` | `nn.Parameter`, clamp [-2,3] |
| `μ` | `self.mu_param` | `nn.Parameter`, clamp [0,20] |
| `dist(vₜ,vⱼ)` | `dist_to_nodes` | `[B,N+1]` |
| `x_curr,y_curr` | `current_coords` | `[B,2]` |
| `kNN(j)` | `knn_indices` | `[B,N+1,k]` |
| `V(s_t)` | `value` | `[B]` |
| `A_t` | `advantages` | `[B,T]` |
