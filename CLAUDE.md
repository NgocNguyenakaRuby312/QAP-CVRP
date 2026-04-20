# CLAUDE.md — QAP-DRL Implementation Reference

**Project:** Quantum-Amplitude Perturbation Deep Reinforcement Learning for Clustered VRP
**Thesis:** "QAP-DRL for Clustered Vehicle Routing Problems" — IU HCMC, 2025
**Target folder:** `cvrp-ppo/` — ALL implementation goes here
**Language:** Python 3.10+, PyTorch ≥ 2.0 (CUDA build — see Section 16)
**Last updated:** April 2026

> **Golden rule:** Implement exactly as specified below. No improvements, no extras,
> no quantum libraries (PennyLane, Qiskit, etc.). This is quantum-INSPIRED classical code.

---

## 1. What This Project Is

A **constructive** DRL solver for CVRP. The agent builds routes from scratch by
selecting the next customer at each step. It is NOT an improvement solver (no 2-opt,
no solution refinement). It is NOT transformer-based. It is NOT using real quantum hardware.

**Two machines:**
- Machine A: `C:\Users\ASUS\Downloads\Thesis Code QAP_VRP\cvrp-ppo\`
- Machine B: `D:\Coding\RubyThesis\QAP-CVRP\cvrp-ppo\`

**ALWAYS run from inside cvrp-ppo/:**
```
cd D:\Coding\RubyThesis\QAP-CVRP\cvrp-ppo
python train_n20.py
```
Running from a parent directory breaks dataset path resolution (`__file__` becomes relative).

---

## 2. Pipeline (7 Stages)

```
CVRP Input [coords, demands, capacity]
    ↓
[1] K-Means Clustering → K sub-problems   (optional, for N≥100)
    ↓
[2] Feature Construction → x_i ∈ ℝ⁵      [d/C, dist, x, y, angle/π]
    ↓
[3] Amplitude Projection → ψ_i ∈ ℝ²      Linear(5→2) + L2Norm  ‖ψ‖=1
    ↓
[4] Per-Node Rotation → ψ'_i ∈ ℝ²        MLP(5→16→1,tanh)→θ → R(θ)·ψ  ‖ψ'‖=1
    ↓  (encoder output cached, does NOT change during decoding)
[5] Context → Query                        [ψ'_curr, cap/C, t/N] → W_q(4→2) → q ∈ ℝ²
    ↓  ↑ repeat N times
[6] Hybrid Scoring + Mask + Sample         Score(j) = q·ψ'_j + λ·Σ_kNN(ψ'_i·ψ'_j)
    ↓
[7] PPO Update                             R = -TotalDistance, GAE advantages, K=3 epochs
    ↓
Optimised Routes
```

**Ablation variant (b):** Steps [3] and [4] replaced by `Linear(5→2) + ReLU` (no norm, no rotation).
Activated via `QAPPolicy(encoder_type="baseline")`. Run via `train_ablation_n20.py`.

---

## 3. Tensor Shapes (cheat sheet)

```
INPUT
  coords:              [B, N+1, 2]      depot at index 0
  demands:             [B, N+1]         demands[:,0] = 0  (depot)
  capacity:            scalar float

ENCODER — QAP mode (default)
  features:            [B, N+1, 5]      order: [d/C, dist, x, y, angle/π]
  psi (projected):     [B, N+1, 2]      ← MUST be unit norm
  theta (from MLP):    [B, N+1]
  psi_prime (rotated): [B, N+1, 2]      ← MUST be unit norm

ENCODER — baseline mode (ablation)
  embedding:           [B, N+1, 2]      ← NOT unit norm (ReLU output, 12 params)

KNN  (precomputed once per instance, reused every decode step)
  knn_indices:         [B, N+1, k]      k=10 for CVRP-20, k=5 for CVRP-50/100

DECODER (per step t)
  psi_curr:            [B, 2]           zero vector if at depot
  context:             [B, 4]           [ψ'_curr(2), cap/C(1), t/N(1)]
  query:               [B, 2]
  context_scores:      [B, N+1]         q · ψ'_j
  interference:        [B, N+1]         Σ_kNN ψ'_i · ψ'_j
  scores:              [B, N+1]         context_scores + λ·interference
  log_probs:           [B, N+1]         log_softmax(masked scores)
  action:              [B]

CRITIC (psi_prime DETACHED before this)
  pooled:              [B, 2]           mean(psi_prime, dim=1)
  value:               [B]

PPO
  advantages:          [B, T]
  returns:             [B, T]
  old_log_probs:       [B, T]
```

---

## 4. Canonical Math

### Feature Construction  (§3.X.3)
```
x_i = [d_i/C,  dist(i,depot),  x_i,  y_i,  atan2(Δy,Δx)/π]
        [0]         [1]          [2]   [3]        [4]
```
- Feature [4] in range [-1, 1]. Assert: `features[:,:,4].abs().max() <= 1.0`
- Depot demand ratio = 0. Assert: `features[:,0,0] == 0`

### Amplitude Projection  (§3.X.4)
```
[α_i, β_i]^T = Normalize(W·x_i + b)     W ∈ ℝ^{2×5}, b ∈ ℝ^2,  α²+β²=1
```
Code: `F.normalize(self.proj(x), p=2, dim=-1)`

### Rotation  (§3.X.5)
```
θ_i = MLP(x_i)      MLP: 5 → 16 → 1, tanh activation
R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]
|ψ'_i⟩ = R(θ_i) · |ψ_i⟩        (norm preserved — no re-normalisation)
```

### Context Query  (§3.X.6)
```
ctx = [ψ'_curr(2D), cap/C(1D), t/N(1D)]  ∈ ℝ^4
q   = W_q · ctx          W_q ∈ ℝ^{2×4}, NO bias
```
ψ'_curr = **zero vector** when at depot (index 0).

### Hybrid Scoring  (§3.X.7)
```
Score(j) = q·ψ'_j  +  λ · Σ_{i∈kNN(j)} (ψ'_i · ψ'_j)
P(j)     = softmax(Score(j))   [infeasible masked to -1e9 BEFORE softmax]
λ = nn.Parameter(torch.tensor(0.1))   ← learnable
```

### PPO Loss  (§3.X.8)
```
r_t     = π_new(a|s) / π_old(a|s)
L_clip  = -E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]
L_value = MSE(V(s_t), R_t)   [psi_prime DETACHED]
L_ent   = -H[π(·|s_t)]
L_total = L_clip + c1·L_value + c2·L_ent
```

---

## 5. Parameter Budget

| Component          | Spec          | Params |
|--------------------|--------------|--------|
| W (amplitude proj) | 2×5          | 10     |
| b (proj bias)      | 2×1          | 2      |
| MLP rotation       | 5→16→1, tanh | ~113   |
| W_q (query proj)   | 2×4, no bias | 8      |
| λ (balance scalar) | scalar       | 1      |
| Critic MLP         | 2→64→1       | ~257   |
| **Actor total**    |              | **~134** |
| **Full total**     |              | **~391** |
| **Baseline total** | (no proj+rot)| **~278** |

---

## 6. Current Hyperparameters — Phase 1b (v8)

```python
# Problem
graph_size   : [20, 50, 100]
capacity     : {20: 30, 50: 40, 100: 50}   # Kool et al. 2019

# Architecture
embedding_dim : 2       hidden_dim : 16
knn_k         : 10      # Phase 1: was 5 (CVRP-20); use 5 for CVRP-50/100
lambda_init   : 0.1

# PPO
K_epochs  : 3       eps_clip  : 0.2     gamma     : 0.99
gae_lambda: 0.95    c1        : 0.5     c2        : 0.01  # thesis spec — NOT 0.05

# Training
lr            : 1e-4
eta_min       : 1e-5          # v4 fix: was 1e-6 — caused training freeze
batch_size    : 512           # Phase 1b: was 256 — stronger advantage signal
epoch_size    : 128_000       # thesis spec — was 51,200
n_epochs      : 200
max_grad_norm : 1.0
aug_samples   : 8             # inference augmentation (evaluate_augmented)

# Auto-calculated
batches_per_epoch : 250       # 128,000 ÷ 512
total_opt_steps   : 1_200_000 # 200 × 250 × 3 × 8

# Evaluation
val_size        : 500         # VAL_EVAL_SIZE
decode_strategy : 'augmented' # evaluate_augmented(n_samples=8)
seed            : 1234
```

---

## 7. Data Generation

```python
coords  = torch.FloatTensor(B, N+1, 2).uniform_(0, 1)   # ALL random (depot too)
demands = torch.zeros(B, N+1, dtype=torch.long)
demands[:, 1:] = torch.randint(1, 10, (B, N))           # depot demand = 0
capacity = {20: 30, 50: 40, 100: 50}[N]
# Move to device after generation: coords.to(device), demands.to(device)
```

---

## 8. File Structure (current)

```
THESIS CODE QAP_VRP/
├── CLAUDE.md
├── cvrp-ppo/
│   ├── run.py
│   ├── options.py
│   ├── train_n20.py              ← v8 Phase 1b [ACTIVE]
│   ├── train_n50.py              ← v8 Phase 1b [ACTIVE]
│   ├── train_n100.py
│   ├── train_n10.py
│   ├── train_ablation_n20.py     ← NEW: ablation study
│   ├── encoder/
│   │   ├── feature_constructor.py
│   │   ├── amplitude_projection.py
│   │   ├── rotation_mlp.py
│   │   ├── rotation.py
│   │   ├── qap_encoder.py
│   │   └── baseline_encoder.py   ← NEW: pure DRL baseline
│   ├── decoder/
│   ├── environment/
│   ├── models/
│   │   └── qap_policy.py         ← UPDATED: encoder_type param
│   ├── training/
│   │   ├── ppo_agent.py          ← v4: eta_min=1e-5
│   │   ├── rollout_buffer.py
│   │   └── evaluate.py           ← UPDATED: evaluate_augmented()
│   ├── utils/
│   │   ├── knn.py
│   │   ├── data_generator.py
│   │   ├── ortools_refs.py       ← UPDATED: rich banner
│   │   └── ortools_solver.py     ← UPDATED: percentiles + timing
│   ├── datasets/
│   │   ├── val_n20.pkl
│   │   ├── val_n50.pkl
│   │   ├── val_n100.pkl
│   │   └── ortools_refs.json
│   └── outputs/
│       ├── n20/
│       ├── n50/
│       └── ablation_n20/         ← NEW
└── ref_code/                     ← READ-ONLY
```

---

## 9. Implementation Order (for new components)

**Phases 1–4 are COMPLETE.** Current work:

1. Run Phase 1b training: `python train_n20.py` (from inside cvrp-ppo/)
2. Run ablation: `python train_ablation_n20.py`
3. If Phase 1b gap < 15%: proceed to Phase 2 (amplitude dim 2→4)
4. Phase 3: 400 epochs + CosineAnnealingWarmRestarts

---

## 10. Critical Rules

### ALWAYS
- `F.normalize(..., p=2, dim=-1)` after amplitude projection (QAP mode)
- `scores[mask] = -1e9` BEFORE `F.log_softmax`
- `mask[:, 0] = False` — depot NEVER masked
- Diagonal = `inf` before kNN topk (no self-loops)
- Shape comment on every tensor op
- `Categorical.sample()` + `.log_prob()` during training
- Every tensor and model `.to(device)`
- `psi_prime.detach()` before critic head in ppo_agent
- `import matplotlib.ticker` in all train files (chart rendering)

### NEVER
- Quantum libraries (PennyLane, Qiskit)
- Mask after softmax
- `argmax` during training
- Re-normalise after rotation
- Hardcode `"cuda"` or `"cpu"`
- 2-opt, curriculum learning
- Run from parent directory (path bug)

---

## 11. Top Pitfalls

| # | Pitfall | Fix |
|---|---------|-----|
| P1 | Missing L2 norm | `F.normalize(..., p=2, dim=-1)` — QAP mode only |
| P2 | Mask after softmax | -1e9 BEFORE log_softmax |
| P3 | Depot masked | `mask[:, 0] = False` |
| P4 | kNN self-loops | `diagonal.fill_(inf)` before topk |
| P5 | NaN in rotation | `torch.clamp(theta, -10, 10)` |
| P6 | Capacity not reset | `if action==0: cap = capacity` |
| P7 | Depot demand ≠ 0 | `demands[:, 0] = 0` |
| P8 | Wrong feature order | `[d/C, dist, x, y, angle/π]` |
| P9 | Angle not /π | `atan2(...) / math.pi` |
| P10 | kNN stale after cluster | Recompute kNN per sub-problem |
| P11 | argmax during training | `Categorical.sample()` |
| P12 | λ as float | `nn.Parameter(torch.tensor(0.1))` |
| P13 | Tensor on wrong device | `.to(device)` everywhere |
| P14 | OOM RTX 3050 | batch_size=256; `torch.cuda.empty_cache()` |
| P15 | Path error (FileNotFoundError) | cd into cvrp-ppo/ before running |
| P16 | Chart tick overflow | `import matplotlib.ticker`; explicit `set_ylim` on twinx axes |
| P17 | Advantage collapse (clip≈0) | Reduce entropy_coef; do not exceed 0.02 |

---

## 12. Validation Checklist

```python
# Encoder (QAP mode)
assert features.shape == (B, N+1, 5)
assert psi_prime.shape == (B, N+1, 2)
assert features[:, 0, 0].sum() == 0,       "depot demand ratio must be 0"
assert features[:,:,4].abs().max() <= 1.0,  "angle must be in [-1,1]"
norms = psi_prime.norm(dim=-1)
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "unit norm violated"

# Decoder
assert mask[:, 0].sum() == 0,  "depot must never be masked"

# Tours
for tour in tours:
    customers = [n for n in tour if n != 0]
    assert len(customers) == N and len(set(customers)) == N
    cap = capacity
    for node in tour:
        if node == 0: cap = capacity
        else: cap -= demands[node]; assert cap >= 0

# PPO — CRITICAL: check clip_fraction
ratio = (new_log_probs - old_log_probs).exp()
assert (ratio > 0).all() and ratio.max() < 100
clip_frac = ((ratio < 0.8) | (ratio > 1.2)).float().mean()
assert clip_frac > 0.001, "Advantage signal collapsed — reduce entropy_coef"

# Device
assert next(model.parameters()).device.type == device.type
```

---

## 13. Smoke Test

```python
model = QAPPolicy(knn_k=10, encoder_type="qap").to(device)
opt   = Adam(model.parameters(), lr=1e-3)
inst  = generate_instance(B=2, N=5, capacity=15, device=device)
r_start = evaluate(model, inst, greedy=True).mean().item()
for _ in range(20):
    train_step(model, opt, inst)
r_end = evaluate(model, inst, greedy=True).mean().item()
assert r_end > r_start, f"no improvement: {r_start:.3f} → {r_end:.3f}"
```

---

## 14. Key Equations

```
x_i      = [d_i/C, ‖i-depot‖, x_i, y_i, atan2(Δy,Δx)/π]
ψ_i      = normalize(W·x_i + b),  ‖ψ_i‖=1      (QAP mode)
θ_i      = tanh_MLP(5→16→1)(x_i)               (QAP mode)
ψ'_i     = R(θ_i)·ψ_i,  ‖ψ'_i‖=1              (QAP mode)
embed_i  = ReLU(W·x_i),  no norm               (baseline mode)
ctx_t    = [ψ'_curr, cap_t/C, t/N]
q_t      = W_q · ctx_t
Score(j) = q_t·ψ'_j + λ·Σ_{kNN}(ψ'_i·ψ'_j)
L        = L_clip + c1·L_value + c2·L_entropy
```

---

## 15. Glossary

| Thesis | Code | Shape |
|--------|------|-------|
| `x_i` | `features` | `[B,N+1,5]` |
| `ψ_i` | `psi` | `[B,N+1,2]` |
| `θ_i` | `theta` | `[B,N+1]` |
| `ψ'_i` | `psi_prime` | `[B,N+1,2]` |
| `q_t` | `query` | `[B,2]` |
| `Score(j)` | `scores` | `[B,N+1]` |
| `λ` | `self.lambda_param` | `nn.Parameter` |
| `kNN(j)` | `knn_indices` | `[B,N+1,k]` |
| `V(s_t)` | `value` | `[B]` |
| `A_t` | `advantages` | `[B,T]` |

---

## 16. GPU / CUDA (Hardware: NVIDIA GeForce RTX 3050, 4GB VRAM, CUDA 13.2)

### Verify First
```python
import torch
print(torch.cuda.is_available())           # must be True
print(torch.cuda.get_device_name(0))       # NVIDIA GeForce RTX 3050
```

### VRAM Budget

| Problem | batch_size | VRAM | Status |
|---------|-----------|------|--------|
| CVRP-20 | 512 | ~1.0 GB | ✓ Safe (Phase 1b) |
| CVRP-50 | 512 | ~2.0 GB | ✓ Safe |
| CVRP-100 | 256 | ~2.0 GB | ✓ Safe |
| CVRP-100 | 512 | ~4.0 GB | ⚠ Borderline |

### Training Time Estimates

| Problem | Time/epoch | 200 epochs |
|---------|-----------|------------|
| CVRP-20 (B=512) | ~4–6 min | ~13–20 hrs |
| CVRP-50 (B=512) | ~10–15 min | ~33–50 hrs |

---

## 17. OR-Tools Reference

Runs automatically via `ensure_ortools_ref()` before training. Cached in `datasets/ortools_refs.json`.

**CVRP-20 (cached):**
- mean_tour = 6.1915
- std_tour = 0.8048
- 5% target = ≤ 6.5011 (this is the number the model must beat)

Percentile fields (p10–p90) and timing stats added in ortools_solver.py v2 — will populate on next fresh run.

---

## 18. Training Results History (CVRP-20)

| Run | entropy_coef | batch | eta_min | Best tour | Gap | Key finding |
|-----|-------------|-------|---------|-----------|-----|-------------|
| Run 2 | 0.01 | 256 | 1e-6 | 7.640 | 23.4% | LR froze at ep160 |
| Run 3 | 0.02 | 256 | 1e-6 | 7.685 | 24.1% | LR froze |
| Run 4 | 0.02 | 256 | 1e-6 | 7.679 | 24.0% | LR froze |
| Run 5 | 0.02 | 256 | **1e-5** | 7.668 | 23.8% | LR fix confirmed |
| Phase 1 | 0.05 | 256 | 1e-5 | 7.228 | 17.1% | adv collapse ep50 |
| Phase 1b | **0.01** | **512** | 1e-5 | pending | target <15% | — |

**Key lessons learned:**
1. `eta_min=1e-6` → LR decays to near-zero → policy freezes (all runs before Run 5)
2. `entropy_coef=0.05` → entropy stays at 0.55 (good) but adv_std → 0.69, clip_frac → 0.014% (bad)
3. Root cause of stall: not entropy collapse but advantage signal collapse from over-exploration
4. Fix: thesis spec `entropy_coef=0.01` + `batch_size=512` for stronger gradients
