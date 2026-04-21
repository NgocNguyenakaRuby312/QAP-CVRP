# CLAUDE.md — QAP-DRL Implementation Reference

**Project:** Quantum-Amplitude Perturbation Deep Reinforcement Learning for Clustered VRP
**Thesis:** "QAP-DRL for Clustered Vehicle Routing Problems" — IU HCMC, 2025
**Target folder:** `cvrp-ppo/` — ALL implementation goes here
**Language:** Python 3.10+, PyTorch ≥ 2.0 (CUDA build — see Section 16)
**Last updated:** May 2026

**KEY REFERENCE PAPER:**
> **Giang et al. (2025)** — *"Vehicle Routing Problems via Quantum Graph Attention Network
> Deep Reinforcement Learning"*
>
> Q-GAT uses real PQCs via PennyLane inside a GAT encoder for VRP, trained with PPO.
> QAP-DRL replaces PQCs with purely classical quantum-inspired operations.

---

## 0. Methodology Changes (May 2026) — Changes 1+2

Two changes fix proximity-blindness in the decoder scoring.

### Change 1 — Distance Proximity Penalty (§3.3.4)
Score formula gains a third term:
```
Score(j) = q · ψ'ⱼ  +  λ · E_kNN(j)  −  μ · dist(vₜ, vⱼ)
```
- `μ` = learnable scalar `nn.Parameter` (init 0.5)
- Implemented in: `decoder/hybrid_scoring.py` (`self.mu_param`)
- +1 parameter

### Change 2 — Spatial Context Grounding (§3.3.3)
Context vector expands from ℝ⁴ → ℝ⁶:
```
ctx = [ψ'_curr(2), cap/C(1), t/N(1), x_curr(1), y_curr(1)]   W_q ∈ ℝ^{2×6}
```
- Implemented in: `decoder/context_query.py` (`self.Wq` is now 2×6)
- Returns `(query [B,2], current_coords [B,2])` — always unpack both
- +4 parameters

### Change 3 — Dynamic Proximity Feature (§3.3.1)
Not implemented. The encoder is STATIC (5D features, computed once).
Spatial awareness is provided by Changes 1+2 in the decoder.

---

## 1. What This Project Is

A **constructive** DRL solver for CVRP. NOT improvement, NOT transformer, NOT quantum hardware.

**Two machines:**
- Machine A: `C:\Users\ASUS\Downloads\Thesis Code QAP_VRP\cvrp-ppo\`
- Machine B: `D:\Coding\RubyThesis\QAP-CVRP\cvrp-ppo\`

**ALWAYS run from inside cvrp-ppo/.**

---

## 2. Pipeline (7 Stages)

```
CVRP Input [coords, demands, capacity]
    ↓
[1] K-Means Clustering → K sub-problems   (optional, N≥100)
    ↓
[2] Feature Construction → xᵢ ∈ ℝ⁵     [d/C, dist_depot, x, y, angle/π]   (STATIC)
    ↓
[3] Amplitude Projection → ψᵢ ∈ ℝ²       Linear(5→2) + L2Norm  ‖ψ‖=1
    ↓
[4] Per-Node Rotation → ψ'ᵢ ∈ ℝ²         MLP(5→16→1,tanh)→θ → R(θ)·ψ  ‖ψ'‖=1
    ↓  (encoder runs ONCE — psi_prime fixed for all decode steps)
[5] Context → Query                        [ψ'_curr, cap/C, t/N, x_curr, y_curr] → W_q(6→2) → q  ← Change 2
    ↓  ↑ repeat N times
[6] Hybrid Scoring + Mask + Sample         Score(j) = q·ψ'_j + λ·E_kNN(j) − μ·dist(vₜ,vⱼ)  ← Change 1
    ↓
[7] PPO Update                             R = -TotalDistance, GAE advantages, K=3 epochs
```

---

## 3. Tensor Shapes (cheat sheet)

```
INPUT
  coords:              [B, N+1, 2]      depot at index 0
  demands:             [B, N+1]         demands[:,0] = 0  (depot)
  capacity:            scalar float

ENCODER — QAP mode (default, STATIC — runs once)
  features:            [B, N+1, 5]      [d/C, dist_depot, x, y, angle/π]
  psi (projected):     [B, N+1, 2]      ← MUST be unit norm
  theta (from MLP):    [B, N+1]
  psi_prime (rotated): [B, N+1, 2]      ← MUST be unit norm, FIXED for all steps

ENCODER — baseline mode (ablation)
  features:            [B, N+1, 5]      same features
  embedding:           [B, N+1, 2]      ← NOT unit norm (ReLU output)

KNN  (precomputed once per instance, from spatial coords — not re-done per step)
  knn_indices:         [B, N+1, k]

DECODER (per step t)
  psi_curr:            [B, 2]           zero vector if at depot
  context:             [B, 6]           [ψ'_curr(2), cap/C(1), t/N(1), x_curr(1), y_curr(1)]
  query:               [B, 2]
  current_coords:      [B, 2]           returned by context_query
  context_scores:      [B, N+1]         q · ψ'_j
  interference:        [B, N+1]         Σ_kNN ψ'_i · ψ'_j
  dist_to_nodes:       [B, N+1]         ‖coords_j − current_coords‖₂
  scores:              [B, N+1]         context_scores + λ·interference − μ·dist_to_nodes
  log_probs:           [B, N+1]         log_softmax(masked scores)
  action:              [B]

evaluate_actions() PPO path
  cur_coords_3d:       [mb, T, 2]       vehicle coords per step (for dist penalty)
  psi_prime_3d:        [mb, T, N+1, 2]  BROADCAST from static psi_prime
  dist_to_nodes:       [mb, T, N+1]
  scores:              [mb, T, N+1]
```

---

## 4. Canonical Math

### Feature Construction  (§3.X.3)
```
xᵢ = [d_i/C,  dist(i,depot),  x_i,  y_i,  atan2(Δy,Δx)/π]
      [0]         [1]          [2]   [3]        [4]
```
- All 5 features are STATIC — computed once before decoding

### Amplitude Projection  (§3.X.4)
```
[α_i, β_i]^T = Normalize(W·x_i + b)     W ∈ ℝ^{2×5}, b ∈ ℝ^2
```

### Rotation  (§3.X.5)
```
θ_i = MLP(x_i)      MLP: 5 → 16 → 1, tanh
ψ'_i = R(θ_i) · ψ_i
```

### Context Query  (§3.X.6) — Change 2
```
ctx = [ψ'_curr(2D), cap/C(1D), t/N(1D), x_curr(1D), y_curr(1D)]  ∈ ℝ⁶
q   = W_q · ctx          W_q ∈ ℝ^{2×6}, NO bias
```

### Hybrid Scoring  (§3.X.7) — Change 1
```
Score(j) = q·ψ'_j  +  λ · Σ_{i∈kNN(j)} (ψ'_i · ψ'_j)  −  μ · dist(vₜ, vⱼ)
λ = nn.Parameter(0.1);   μ = nn.Parameter(0.5)
```

---

## 5. Parameter Budget

| Component | Spec | Params |
|---|---|---|
| W (amplitude proj) | 2×5 | 10 |
| b (proj bias) | 2×1 | 2 |
| MLP rotation | 5→16→1 | ~113 |
| W_q (query proj) | **2×6**, no bias | **12** (+4 Change 2) |
| λ (interference) | scalar | 1 |
| μ (dist penalty) | **scalar** | **1** (+1 Change 1) |
| Critic MLP | 2→64→1 | ~257 |
| **Actor total** | | **~139** |
| **Full total** | | **~396** |

---

## 6. Current Hyperparameters — Phase 1b (v9)

```python
BATCH_SIZE   = 512;  EPOCH_SIZE  = 128_000;  N_EPOCHS = 200
LR           = 1e-4; ENTROPY_COEF= 0.01;     KNN_K    = 10
MU_INIT      = 0.5;  LAMBDA_INIT = 0.1;      AUG_SAMPLES = 8
VAL_EVAL_SIZE= 10_000;  ORTOOLS_EVAL_SIZE = 1_000
```

---

## 7. File Status

| File | Change | Status |
|------|--------|--------|
| `encoder/feature_constructor.py` | Static 5D features | ✓ Done |
| `encoder/amplitude_projection.py` | input_dim=5, W 2×5 | ✓ Done |
| `encoder/rotation_mlp.py` | input_dim=5, 5→16→1 | ✓ Done |
| `encoder/qap_encoder.py` | input_dim=5, static forward(state) | ✓ Done |
| `decoder/context_query.py` | Change 2: ctx ℝ⁴→ℝ⁶, Wq 2×4→2×6 | ✓ Done |
| `decoder/hybrid_scoring.py` | Change 1: +mu_param, −μ·dist | ✓ Done |
| `decoder/qap_decoder.py` | C1+C2: current_coords to scoring, no encoder arg | ✓ Done |
| `models/qap_policy.py` | C1+C2: feature_dim=5, broadcast psi_prime | ✓ Done |
| `training/ppo_agent.py` | C1: mu_val in diag | ✓ Done |
| `train_n20.py` | C1+C2: mu_val logged, chart panel 8 updated | ✓ Done |

---

## 8. Critical Rules

### ALWAYS
- `F.normalize(..., p=2, dim=-1)` after amplitude projection (QAP mode)
- `scores[mask] = -1e9` BEFORE `F.log_softmax`
- `mask[:, 0] = False` — depot NEVER masked
- Diagonal = `inf` before kNN topk (no self-loops)
- `context_query.forward()` returns tuple — always unpack: `query, current_coords = ...`
- `decoder.rollout()` uses fixed psi_prime — no per-step re-encoding
- `evaluate_actions()` broadcasts static psi_prime across T steps
- `evaluate_augmented()` uses coordinate augmentation + greedy (NOT stochastic) — evaluate.py v3
- After `plot_route_map(best_route.png)`: run `solve_one_with_routes()` on same instance → `ortools_route.png`
- All `twinx()` axes: `.set_zorder(-1)`, `.patch.set_visible(False)`, `.grid(False)`

### NEVER
- Quantum libraries (PennyLane, Qiskit)
- Mask after softmax
- feature_dim set to 6
- Revert context_dim back to 4 (Change 2 is permanent)
- Remove μ parameter (Change 1 is permanent)
- Run from parent directory (path bug)

---

## 9. Pitfalls (Changes 1+2)

| # | Pitfall | Fix |
|---|---------|-----|
| P15 | context_query returns tuple not tensor | Unpack: `query, current_coords = context_query(...)` |
| P16 | mu_param missing or not nn.Parameter | `self.mu_param = nn.Parameter(torch.tensor(0.5))` |
| P17 | feature_dim set to 6 | Must be 5. Encoder is static 5D |
| P18 | Per-step re-encoding attempted | NEVER re-encode. psi_prime is fixed after initial encode |
| P19 | evaluate_actions rebuilds psi_prime per step | Must BROADCAST static psi_prime — NOT rebuild |
| P24 | Dense grey gridlines on twinx panels | `.set_zorder(-1)`, `.patch.set_visible(False)`, `.grid(False)` |

---

## 10. Validation Checklist

```python
# Encoder (QAP mode)
assert features.shape == (B, N+1, 5),     "features must be 5D"
assert psi_prime.shape == (B, N+1, 2)
norms = psi_prime.norm(dim=-1)
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "unit norm"

# Context (Change 2)
assert ctx.shape[-1] == 6,               "context must be 6D (Change 2)"
assert current_coords.shape == (B, 2)

# Scoring (Change 1)
assert hasattr(model.decoder.hybrid, 'mu_param'), "μ missing"
assert dist_to_nodes.shape == (B, N+1)

# Encoder is STATIC — no per-step re-encoding
# evaluate_actions broadcasts psi_prime

# Masking
assert mask[:, 0].sum() == 0,            "depot must never be masked"
```
