# CLAUDE.md — QAP-DRL Implementation Reference

**Project:** Quantum-Amplitude Perturbation Deep Reinforcement Learning for Clustered VRP
**Thesis:** "QAP-DRL for Clustered Vehicle Routing Problems" — IU HCMC, 2025
**Target folder:** `cvrp-ppo/` — ALL implementation goes here
**Language:** Python 3.10+, PyTorch ≥ 2.0 (CUDA build — see Section 16)

> **Golden rule:** Implement exactly as specified below. No improvements, no extras,
> no quantum libraries (PennyLane, Qiskit, etc.). This is quantum-INSPIRED classical code.

---

## 1. What This Project Is

A **constructive** DRL solver for CVRP. The agent builds routes from scratch by
selecting the next customer at each step. It is NOT an improvement solver (no 2-opt,
no solution refinement). It is NOT transformer-based. It is NOT using real quantum hardware.

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

---

## 3. Tensor Shapes (cheat sheet)

```
INPUT
  coords:              [B, N+1, 2]      depot at index 0
  demands:             [B, N+1]         demands[:,0] = 0  (depot)
  capacity:            scalar float

ENCODER
  features:            [B, N+1, 5]      order: [d/C, dist, x, y, angle/π]
  psi (projected):     [B, N+1, 2]      ← MUST be unit norm
  theta (from MLP):    [B, N+1]
  psi_prime (rotated): [B, N+1, 2]      ← MUST be unit norm

KNN  (precomputed once per instance, reused every decode step)
  knn_indices:         [B, N+1, k]      k=5, no self-loops

DECODER (per step t)
  psi_curr:            [B, 2]           zero vector if at depot
  context:             [B, 4]           [ψ'_curr(2), cap/C(1), t/N(1)]
  query:               [B, 2]
  context_scores:      [B, N+1]         q · ψ'_j
  interference:        [B, N+1]         Σ_kNN ψ'_i · ψ'_j
  scores:              [B, N+1]         context_scores + λ·interference
  log_probs:           [B, N+1]         log_softmax(masked scores)
  action:              [B]

CRITIC
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
L_value = MSE(V(s_t), R_t)
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

---

## 6. Hyperparameters (locked)

```python
# Problem
graph_size   : [20, 50, 100]
capacity     : {20: 30, 50: 40, 100: 50}   # Kool et al. 2019

# Architecture
embedding_dim : 2       hidden_dim : 16     knn_k      : 5
lambda_init   : 0.1

# PPO
K_epochs  : 3       eps_clip  : 0.2     gamma     : 0.99
gae_lambda: 0.95    c1        : 0.5     c2        : 0.01

# Training
lr            : 1e-4
batch_size    : 256     # ← RTX 3050 4GB default; raise to 512 only if no OOM
epoch_size    : 128_000
n_epochs      : 100
max_grad_norm : 1.0

# Evaluation
val_size        : 10_000
decode_strategy : 'greedy'
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

## 8. File Structure

```
THESIS CODE QAP_VRP/
├── CLAUDE.md
├── cvrp-ppo/
│   ├── run.py
│   ├── options.py
│   ├── encoder/
│   │   ├── feature_constructor.py     [d/C, dist, x, y, angle/π]
│   │   ├── amplitude_projection.py    Linear(5→2) + L2 norm
│   │   ├── rotation_mlp.py            MLP(5→16→1, tanh)
│   │   ├── rotation.py                R(θ)·ψ
│   │   └── qap_encoder.py             combines all 4
│   ├── decoder/
│   │   ├── context_query.py           W_q(4→2, no bias)
│   │   ├── hybrid_scoring.py          context + kNN + masking
│   │   └── qap_decoder.py             autoregressive loop
│   ├── environment/
│   │   ├── cvrp_env.py
│   │   └── state.py
│   ├── models/
│   │   └── qap_policy.py              actor + critic (shared encoder)
│   ├── training/
│   │   ├── ppo_agent.py
│   │   ├── rollout_buffer.py
│   │   └── evaluate.py
│   ├── utils/
│   │   ├── knn.py                     spatial coords, no self-loops
│   │   ├── clustering.py              K-Means (N≥100)
│   │   ├── data_generator.py
│   │   ├── seed.py
│   │   ├── logger.py
│   │   ├── checkpoint.py
│   │   └── metrics.py
│   ├── configs/default.yaml
│   ├── datasets/
│   ├── outputs/
│   └── tests/
│       ├── test_env.py
│       ├── test_encoder.py
│       ├── test_decoder.py
│       └── test_smoke.py
└── ref_code/                          ← READ-ONLY
```

---

## 9. Implementation Order

**Phase 1 — Foundation:** seed → data_generator → state → cvrp_env → test_env
**Phase 2 — Encoder:** feature_constructor → amplitude_projection → rotation_mlp → rotation → qap_encoder → test_encoder
**Phase 3 — Decoder:** knn → context_query → hybrid_scoring → qap_decoder → test_decoder
**Phase 4 — Training:** qap_policy → rollout_buffer → ppo_agent → evaluate → options → run → test_smoke
**Phase 5 — Extensions:** clustering

Do NOT move to next phase until the phase test passes.

---

## 10. Critical Rules

### ALWAYS
- `F.normalize(..., p=2, dim=-1)` after amplitude projection
- `scores[mask] = -1e9` BEFORE `F.log_softmax`
- `mask[:, 0] = False` — depot NEVER masked
- Diagonal = `inf` before kNN topk (no self-loops)
- Shape comment on every tensor op
- `Categorical.sample()` + `.log_prob()` during training
- Every tensor and model `.to(device)`

### NEVER
- Quantum libraries (PennyLane, Qiskit)
- Mask after softmax
- `argmax` during training
- Re-normalise after rotation
- Hardcode `"cuda"` or `"cpu"`
- 2-opt, curriculum learning, inference augmentation

---

## 11. Top Pitfalls

| # | Pitfall | Fix |
|---|---------|-----|
| P1 | Missing L2 norm | `F.normalize(..., p=2, dim=-1)` |
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
| P14 | OOM RTX 3050 | batch_size=128; `torch.cuda.empty_cache()` |

---

## 12. Validation Checklist

```python
# Encoder
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

# PPO
ratio = (new_log_probs - old_log_probs).exp()
assert (ratio > 0).all() and ratio.max() < 100

# Device
assert next(model.parameters()).device.type == device.type
```

---

## 13. Smoke Test

```python
model = QAPPolicy(opts).to(device)
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
ψ_i      = normalize(W·x_i + b),  ‖ψ_i‖=1
θ_i      = tanh_MLP(5→16→1)(x_i)
ψ'_i     = R(θ_i)·ψ_i,  ‖ψ'_i‖=1
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
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # ~4.29 GB
```

If False — reinstall PyTorch with CUDA:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Device Pattern (use in every file)
```python
# run.py — detect ONCE, pass everywhere as argument
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
```

### Per-File Device Rules
| File | Required action |
|------|----------------|
| `run.py` | Detect device, pass to all modules |
| `data_generator.py` | Accept `device` arg; `.to(device)` on all output tensors |
| `qap_policy.py` | `model.to(device)` called externally from run.py |
| `knn.py` | `knn_indices.to(device)` before return |
| `rollout_buffer.py` | All stored tensors `.to(device)` |
| `ppo_agent.py` | `torch.cuda.empty_cache()` at epoch start |
| `evaluate.py` | `torch.no_grad()` block; tensors on device |

### VRAM Budget (RTX 3050, 4GB)

| Problem | batch_size | Est. VRAM | Status |
|---------|-----------|-----------|--------|
| CVRP-20 | 256 | ~0.5 GB | ✓ Safe |
| CVRP-50 | 256 | ~1.0 GB | ✓ Safe |
| CVRP-100 | 256 | ~2.0 GB | ✓ Safe |
| CVRP-100 | 512 | ~4.0 GB | ⚠ Borderline — test first |

**Default: batch_size = 256**

### OOM Recovery
```python
# Step 1: reduce batch
batch_size = 128

# Step 2: clear cache more often
torch.cuda.empty_cache()   # start of every epoch

# Step 3: monitor
used  = torch.cuda.memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {used:.2f} / {total:.2f} GB")
```

### Training Time Estimates

| Problem | Time/epoch | 100 epochs total |
|---------|-----------|-----------------|
| CVRP-20 | ~3–5 min | ~5–8 hrs |
| CVRP-50 | ~8–12 min | ~13–20 hrs |
| CVRP-100 | ~20–30 min | ~33–50 hrs |

Save checkpoints every epoch. Train CVRP-100 overnight.
