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

## 0. Methodology Changes (May 2026) — ALL THREE IMPLEMENTED

Three changes fix proximity-blindness in the original scoring.
**These are now the canonical specification — not optional.**

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
Feature vector expands from ℝ⁵ → ℝ⁶:
```
xᵢ(t) = [d/C, dist_depot, x, y, angle/π, dist(i, vₜ)]   ∈ ℝ⁶
```
- `dist(i, vₜ)` = Euclidean distance from candidate i to vehicle's **current position** at step t
- Dynamic: recomputed at every decoding step as the vehicle moves
- Fallback: when `current_node_coords=None`, feature[5] = dist(i, depot)
- Implemented in: `encoder/feature_constructor.py`, `encoder/amplitude_projection.py` (input_dim 5→6),
  `encoder/rotation_mlp.py` (input_dim 5→6), `encoder/qap_encoder.py` (input_dim 5→6, per-step re-encode)
- `decoder/qap_decoder.py` and `training/ppo_agent.py` re-encode at each rollout step
- `models/qap_policy.py` `evaluate_actions()` builds per-step features with `cur_coords_3d`
- +18 parameters (W: +2, MLP first layer: +16)

**Total new params: +23. Full model: ~391 → ~414.**

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
[2] Feature Construction → xᵢ(t) ∈ ℝ⁶   [d/C, dist_depot, x, y, angle/π, dist_curr]  ← Change 3
    ↓
[3] Amplitude Projection → ψᵢ ∈ ℝ²       Linear(6→2) + L2Norm  ‖ψ‖=1                   ← Change 3
    ↓
[4] Per-Node Rotation → ψ'ᵢ ∈ ℝ²         MLP(6→16→1,tanh)→θ → R(θ)·ψ  ‖ψ'‖=1          ← Change 3
    ↓  (re-encoded EACH decode step with current_node_coords)
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

ENCODER — QAP mode (default)
  current_node_coords: [B, 2]           vehicle position at step t (Change 3)
  features:            [B, N+1, 6]      [d/C, dist, x, y, angle/π, dist_curr]  ← Change 3
  psi (projected):     [B, N+1, 2]      ← MUST be unit norm
  theta (from MLP):    [B, N+1]
  psi_prime (rotated): [B, N+1, 2]      ← MUST be unit norm

ENCODER — baseline mode (ablation)
  features:            [B, N+1, 6]      same shape (dist_curr=dist_depot fallback)
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
  cur_coords_3d:       [mb, T, 2]       vehicle coords per step
  features per step:   [mb, N+1, 6]     built T times with cur_coords_3d[:,t,:]
  psi_prime_3d:        [mb, T, N+1, 2]  per-step amplitudes
  dist_to_nodes:       [mb, T, N+1]
  scores:              [mb, T, N+1]
```

---

## 4. Canonical Math

### Feature Construction  (§3.X.3) — Change 3
```
xᵢ(t) = [d_i/C,  dist(i,depot),  x_i,  y_i,  atan2(Δy,Δx)/π,  dist(i,vₜ)]
          [0]         [1]          [2]   [3]        [4]              [5]
```
- Feature [5] = dynamic, recomputed each step with current vehicle position
- Fallback when current_node_coords=None: feature[5] = feature[1]

### Amplitude Projection  (§3.X.4) — Change 3
```
[α_i, β_i]^T = Normalize(W·x_i + b)     W ∈ ℝ^{2×6}, b ∈ ℝ^2   (was 2×5)
```

### Rotation  (§3.X.5) — Change 3
```
θ_i = MLP(x_i)      MLP: 6 → 16 → 1, tanh   (was 5→16→1)
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
| W (amplitude proj) | **2×6** | **12** (+2 Change 3) |
| b (proj bias) | 2×1 | 2 |
| MLP rotation | **6→16→1** | **~129** (+16 Change 3) |
| W_q (query proj) | **2×6**, no bias | **12** (+4 Change 2) |
| λ (interference) | scalar | 1 |
| μ (dist penalty) | **scalar** | **1** (+1 Change 1) |
| Critic MLP | 2→64→1 | ~257 |
| **Actor total** | | **~157** |
| **Full total** | | **~414** |

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
| `encoder/feature_constructor.py` | Change 3: ℝ⁵→ℝ⁶, dist_curr feature | ✓ Done |
| `encoder/amplitude_projection.py` | Change 3: input_dim 5→6 | ✓ Done |
| `encoder/rotation_mlp.py` | Change 3: input_dim 5→6, 6→16→1 | ✓ Done |
| `encoder/qap_encoder.py` | Change 3: input_dim=6, build_features(), per-step encode | ✓ Done |
| `decoder/context_query.py` | Change 2: ctx ℝ⁴→ℝ⁶, Wq 2×4→2×6 | ✓ Done |
| `decoder/hybrid_scoring.py` | Change 1: +mu_param, −μ·dist | ✓ Done |
| `decoder/qap_decoder.py` | Change 3: encoder arg in rollout(), per-step re-encode | ✓ Done |
| `models/qap_policy.py` | All: feature_dim=6, per-step psi_prime_3d in evaluate_actions | ✓ Done |
| `training/ppo_agent.py` | Change 3: enc_ref in collect_rollout(), mu_val in diag | ✓ Done |
| `train_n20.py` | All: mu_val logged, chart panel 8 updated | ✓ Done |

---

## 8. Critical Rules

### ALWAYS
- `F.normalize(..., p=2, dim=-1)` after amplitude projection (QAP mode)
- `scores[mask] = -1e9` BEFORE `F.log_softmax`
- `mask[:, 0] = False` — depot NEVER masked
- Diagonal = `inf` before kNN topk (no self-loops)
- `context_query.forward()` returns tuple — always unpack: `query, current_coords = ...`
- `decoder.rollout()` receives `encoder=enc_ref` for Change 3 re-encoding
- `evaluate_actions()` builds `psi_prime_3d [mb,T,N+1,2]` per step internally
- `evaluate_augmented()` uses coordinate augmentation + greedy (NOT stochastic) — evaluate.py v3
- After `plot_route_map(best_route.png)`: run `solve_one_with_routes()` on same instance → `ortools_route.png`
- All `twinx()` axes: `.set_zorder(-1)`, `.patch.set_visible(False)`, `.grid(False)`

### NEVER
- Quantum libraries (PennyLane, Qiskit)
- Mask after softmax
- Revert feature_dim back to 5 (Change 3 is permanent)
- Revert context_dim back to 4 (Change 2 is permanent)
- Remove μ parameter (Change 1 is permanent)
- Run from parent directory (path bug)

---

## 9. New Pitfalls (Changes 1+2+3)

| # | Pitfall | Fix |
|---|---------|-----|
| P15 | context_query returns tuple not tensor | Unpack: `query, current_coords = context_query(...)` |
| P16 | mu_param missing or not nn.Parameter | `self.mu_param = nn.Parameter(torch.tensor(0.5))` |
| P17 | feature_dim=5 after Change 3 | `FullEncoder(feature_dim=6)` — all encoder inputs expect 6D |
| P18 | encoder not passed to rollout() | `decoder.rollout(..., encoder=enc_ref)` for QAP mode |
| P19 | evaluate_actions uses broadcast psi_prime | In QAP mode, psi_prime_3d is rebuilt per step — NOT broadcast |
| P20 | FeatureBuilder returns 5D | Check feature_dim=6 in FeatureBuilder; assert `features.shape[-1]==6` |
| P21 | evaluate_augmented stochastic invalid with Change 3 | coord aug + greedy (evaluate.py v3) |
| P22 | val_tour > greedy_tour in log = aug bug NOT overfitting | See P21. After fix, val_tour ≤ greedy_tour |
| P23 | Reported gap inflated ~12% | val_tour from broken aug; real gap = greedy_tour-based |
| P24 | Dense grey gridlines on twinx panels | `.set_zorder(-1)`, `.patch.set_visible(False)`, `.grid(False)` |

---

## 10. Validation Checklist

```python
# Encoder (QAP mode)
assert features.shape == (B, N+1, 6),     "features must be 6D after Change 3"
assert psi_prime.shape == (B, N+1, 2)
norms = psi_prime.norm(dim=-1)
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "unit norm"

# Context (Change 2)
assert ctx.shape[-1] == 6,               "context must be 6D (Change 2)"
assert current_coords.shape == (B, 2)

# Scoring (Change 1)
assert hasattr(model.decoder.hybrid, 'mu_param'), "μ missing"
assert dist_to_nodes.shape == (B, N+1)

# Decoder (Change 3)
# encoder ref must be passed in rollout for QAP mode
# evaluate_actions builds psi_prime_3d [mb,T,N+1,2] per step

# Masking
assert mask[:, 0].sum() == 0,            "depot must never be masked"
```
