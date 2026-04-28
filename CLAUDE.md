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
Context vector: ℝ⁸ = [ψ'_curr(4), cap/C(1), t/N(1), x_curr(1), y_curr(1)]
```
ctx = [ψ'_curr(4), cap/C(1), t/N(1), x_curr(1), y_curr(1)]   W_q ∈ ℝ^{4×8}
```
- Implemented in: `decoder/context_query.py` (`self.Wq` is now 4×8)
- Returns `(query [B,4], current_coords [B,2])` — always unpack both
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
[3] Amplitude Projection → ψᵢ ∈ ℝ⁴       Linear(5→4) + L2Norm  ‖ψ‖=1   (S³ hypersphere)
    ↓
[4] Per-Node Rotation → ψ'ᵢ ∈ ℝ⁴         MLP(5→32→6,tanh)→6 Givens angles → SO(4)·ψ  ‖ψ'‖=1
    ↓  (encoder runs ONCE — psi_prime fixed for all decode steps)
[5] Context → Query                        [ψ'_curr(4), cap/C, t/N, x_curr, y_curr] → W_q(8→4) → q
    ↓  ↑ repeat N times
[6] Hybrid Scoring + Mask + Sample         Score(j) = q·ψ'_j + λ·E_kNN(j) − μ·dist(vₜ,vⱼ)
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
  psi (projected):     [B, N+1, 4]      ← S³ hypersphere, MUST be unit norm
  theta (from MLP):    [B, N+1, 6]      ← 6 Givens angles for SO(4)
  psi_prime (rotated): [B, N+1, 4]      ← MUST be unit norm, FIXED for all steps

ENCODER — baseline mode (ablation)
  features:            [B, N+1, 5]      same features
  embedding:           [B, N+1, 4]      ← NOT unit norm (ReLU output)

KNN  (precomputed once per instance, from spatial coords)
  knn_indices:         [B, N+1, k]

DECODER (per step t)
  psi_curr:            [B, 4]           zero vector if at depot
  context:             [B, 8]           [ψ'_curr(4), cap/C(1), t/N(1), x_curr(1), y_curr(1)]
  query:               [B, 4]
  current_coords:      [B, 2]           returned by context_query
  context_scores:      [B, N+1]         q · ψ'_j
  interference:        [B, N+1]         Σ_kNN ψ'_i · ψ'_j
  dist_to_nodes:       [B, N+1]         ‖coords_j − current_coords‖₂
  scores:              [B, N+1]         context_scores + λ·interference − μ·dist_to_nodes
  log_probs:           [B, N+1]         log_softmax(masked scores)
  action:              [B]

evaluate_actions() PPO path
  cur_coords_3d:       [mb, T, 2]       vehicle coords per step
  psi_prime_3d:        [mb, T, N+1, 4]  BROADCAST from static psi_prime
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
[α_i, β_i, γ_i, δ_i]^T = Normalize(W·x_i + b)     W ∈ ℝ^{4×5}, b ∈ ℝ^4
```

### Rotation  (§3.X.5)
```
θ_i = MLP(x_i)      MLP: 5 → 32 → 6, tanh   (6 Givens angles for SO(4))
ψ'_i = R(Θ_i) · ψ_i   R = G₆·G₅·G₄·G₃·G₂·G₁ on planes (01)(02)(03)(12)(13)(23)
```

### Context Query  (§3.X.6)
```
ctx = [ψ'_curr(4D), cap/C(1D), t/N(1D), x_curr(1D), y_curr(1D)]  ∈ ℝ⁸
q   = W_q · ctx          W_q ∈ ℝ^{4×8}, NO bias
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
| W (amplitude proj) | 4×5 | 20 |
| b (proj bias) | 4×1 | 4 |
| MLP rotation | 5→32→6 | ~230 |
| W_q (query proj) | 4×8, no bias | 32 |
| λ (interference) | scalar | 1 |
| μ (dist penalty) | scalar | 1 |
| Critic MLP | 4→64→1 | ~321 |
| **Actor total** | | **~288** |
| **Full total** | | **~609** |

---

## 6. Current Hyperparameters — Phase 2+

```python
BATCH_SIZE   = 512;  EPOCH_SIZE  = 128_000;  N_EPOCHS = 200
LR           = 1e-4; ENTROPY_COEF= 0.03;     KNN_K    = 10
MU_INIT      = 0.5;  LAMBDA_INIT = 0.1;      AUG_SAMPLES = 8
AMP_DIM      = 4;    HIDDEN_DIM  = 32
VAL_EVAL_SIZE= 10_000;  ORTOOLS_EVAL_SIZE = 1_000
Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-5)
μ clamp: [0, 20];  λ clamp: [-2, 3]
No weight_decay on μ
```

---

## 7. File Status

| File | Change | Status |
|------|--------|--------|
| `encoder/feature_constructor.py` | Static 5D features | ✓ Done |
| `encoder/amplitude_projection.py` | output_dim=4, W 4×5 | ✓ Done |
| `encoder/rotation_mlp.py` | n_angles=6, 5→32→6 | ✓ Done |
| `encoder/rotation.py` | 6 Givens rotations for SO(4) | ✓ Done |
| `encoder/qap_encoder.py` | amp_dim=4, static forward(state) | ✓ Done |
| `decoder/context_query.py` | ctx ℝ⁸, Wq 4×8, embed_dim=4 | ✓ Done |
| `decoder/hybrid_scoring.py` | dimension-agnostic, μ clamp [0,20], λ clamp [-2,3] | ✓ Done |
| `decoder/qap_decoder.py` | context_dim=8, embed_dim=4 | ✓ Done |
| `models/qap_policy.py` | amp_dim=4, context_dim=8, critic 4→64→1 | ✓ Done |
| `training/ppo_agent.py` | WarmRestarts, no weight_decay on μ | ✓ Done |
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
| P20 | Post-training reload missing amp_dim/hidden_dim | Must pass `amp_dim=AMP_DIM, hidden_dim=HIDDEN_DIM` |
| P24 | Dense grey gridlines on twinx panels | `.set_zorder(-1)`, `.patch.set_visible(False)`, `.grid(False)` |

---

## 10. Validation Checklist

```python
# Encoder (QAP mode)
assert features.shape == (B, N+1, 5),     "features must be 5D"
assert psi_prime.shape == (B, N+1, 4),     "4D amplitude vectors"
norms = psi_prime.norm(dim=-1)
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "unit norm"

# Context (Phase 2)
assert ctx.shape[-1] == 8,               "context must be 8D (D+4 where D=4)"
assert current_coords.shape == (B, 2)
assert query.shape == (B, 4),            "query must be 4D"

# Scoring
assert hasattr(model.decoder.hybrid, 'mu_param'), "μ missing"
assert dist_to_nodes.shape == (B, N+1)

# Masking
assert mask[:, 0].sum() == 0,            "depot must never be masked"
```
