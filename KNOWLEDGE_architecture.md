# KNOWLEDGE: QAP-DRL Architecture & Methodology

**Purpose:** Complete mathematical specification and PyTorch implementation reference for every QAP-DRL component.
**When to reference:** Any coding, implementation, or architecture question. This is the single source of truth.
**Last updated:** April 2026

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

### Data Generation (Kool et al. 2019 protocol)

```python
coords_depot = torch.FloatTensor(1, 2).uniform_(0, 1)       # depot
coords_customers = torch.FloatTensor(N, 2).uniform_(0, 1)   # customers
demands = torch.randint(1, 10, (N,))                         # DiscreteUniform(1, 9)
# Training: 128,000 instances/epoch, regenerated each epoch
# Validation: 500 fixed instances (VAL_EVAL_SIZE)
# Test: fixed instances per problem size
```

---

## 2. Architecture Overview

```
Input: coords [B, N+1, 2], demands [B, N+1], capacity C
    │
    ▼
┌─────────────────────── ENCODER ───────────────────────┐
│  ① Feature Construction    → [B, N+1, 5]              │
│  ② Amplitude Projection    Linear(5→2) + L2 norm → [B, N+1, 2]  │
│  ③ Rotation MLP            MLP(5→16→1) → θ_i → [B, N+1]        │
│  ④ Rotation Matrix         R(θ_i) · ψ_i → ψ'_i → [B, N+1, 2]  │
└───────────────────────────────────────────────────────┘
    │ ψ'_i (unit circle embeddings)
    ▼
┌─────────────────────── DECODER (autoregressive) ──────┐
│  For each step t = 0, 1, ..., N-1:                     │
│  ⑤ Context Query   [ψ'_curr, cap/C, t/N] → q  → [B, 2]       │
│  ⑥ Hybrid Scoring  q·ψ'_j + λ·E_kNN(j) → scores → [B, N+1]   │
│  ⑦ Masking + Softmax → action probabilities → [B, N+1]         │
│  ⑧ Sample/Greedy → next node → [B]                             │
└───────────────────────────────────────────────────────┘
    │ complete tour
    ▼
┌─────────────────────── PPO TRAINING ──────────────────┐
│  Actor:  π(a|s) from decoder softmax                   │
│  Critic: MLP on mean-pooled ψ' → V(s) scalar          │
│  Loss:   L_clip + c1·L_value + c2·L_entropy            │
└───────────────────────────────────────────────────────┘
```

### Parameter Budget (from Thesis System Design)

| Component | Specification | Parameters |
|-----------|--------------|------------|
| W (amplitude projection) | 2 × 5 | 10 |
| b (projection bias) | 2 × 1 | 2 |
| MLP (rotation) | 5→16→1 | ~113 |
| W_q (query projection) | 2 × 4, no bias | 8 |
| λ (balance scalar) | 1 | 1 |
| Critic MLP | 2→64→1 | ~257 |
| **Actor total** | | **~134** |
| **Full total** | | **~391** |

### Ablation Baseline (variant b) — Pure DRL, no QAP

| Component | Specification | Parameters |
|-----------|--------------|------------|
| BaselineEncoder | Linear(5→2) + ReLU, no L2 norm, no rotation | 12 |
| W_q, λ, Critic | same as full model | 266 |
| **Baseline total** | | **~278** |

The baseline intentionally has fewer parameters than the full QAP-DRL model — making any quality advantage of QAP-DRL conservative and stronger as evidence.

---

## 3. Encoder Components (Thesis §3.X.3–3.X.5)

### 3.1 Feature Construction (Thesis §3.X.3)

Input: raw coordinates + demands + capacity
Output: `x_i ∈ ℝ^5` for each node i

**Thesis-specified feature order:**
```
x_i = [d_i/C, dist(i, depot), x_i, y_i, α_i/π]
```

```python
def feature_constructor(coords, demands, capacity):
    depot = coords[:, 0:1, :]                              # [B, 1, 2]
    diff = coords - depot                                   # [B, N+1, 2]
    dist_to_depot = torch.norm(diff, dim=-1)                # [B, N+1]
    angle_to_depot = torch.atan2(
        diff[:, :, 1], diff[:, :, 0]
    ) / math.pi                                             # [B, N+1] normalized to [-1, 1]
    demand_ratio = demands / capacity                       # [B, N+1]

    features = torch.stack([
        demand_ratio,              # [0] d_i / C
        dist_to_depot,             # [1] dist(i, depot)
        coords[:, :, 0],          # [2] x coordinate
        coords[:, :, 1],          # [3] y coordinate
        angle_to_depot,            # [4] α_i / π
    ], dim=-1)                                              # [B, N+1, 5]
    return features
```

### 3.2 Amplitude Projection (Thesis §3.X.4)

```
ψ_i = Normalize(W_proj · x_i + b_proj)
where W_proj ∈ ℝ^{2×5}, b_proj ∈ ℝ^2
s.t. α_i² + β_i² = 1
```

**Critical:** L2 normalization is NON-NEGOTIABLE. Every ψ_i must have ||ψ_i|| = 1.

### 3.3 Rotation MLP (Thesis §3.X.5)

```
θ_i = MLP(x_i)   where MLP: ℝ^5 → ℝ^16 → ℝ^1, tanh activation
R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]
ψ'_i = R(θ_i) · ψ_i        (norm preserved — no re-normalization needed)
```

### 3.4 Ablation Baseline Encoder (`encoder/baseline_encoder.py`)

For Ablation Study Tier 2, variant (b): remove amplitude projection + rotation, use plain MLP.

```python
class BaselineEncoder(nn.Module):
    def __init__(self, input_dim=5, output_dim=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)  # 12 params

    def forward(self, features):
        return F.relu(self.proj(features))  # [B, N+1, 2] — NOT unit-norm, no rotation
```

`FullBaselineEncoder` is the drop-in replacement for `FullEncoder` — identical interface,
returns `(embedding, features, knn_indices)`. Used via `QAPPolicy(encoder_type="baseline")`.

---

## 4. Decoder Components (Thesis §3.X.6–3.X.7)

### 4.1 Context Query Construction (Thesis §3.X.6)

```
context_t = [ψ'_{curr}, remaining_cap / C, t / N]  ∈ ℝ^4
q_t = W_q · context_t   where W_q ∈ ℝ^{2×4}, no bias
```
**ψ′_curr = zero vector if at the depot.**

### 4.2 Hybrid Scoring Mechanism (Thesis §3.X.7)

```
S_context(j) = q_t · ψ'_j
E_kNN(j) = Σ_{i ∈ kNN(j)} (ψ'_i · ψ'_j)
Score(j) = S_context(j) + λ · E_kNN(j)
λ is learnable (initialized at 0.1)
```

### 4.3 kNN Precomputation

```python
def compute_knn(coords, k=10):  # k=10 for CVRP-20, Phase 1
    dists = torch.cdist(coords, coords)
    dists.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))   # no self-loops
    _, knn_indices = dists.topk(k, dim=-1, largest=False)
    return knn_indices  # [B, N+1, k]
```

---

## 5. Critic Network

```python
class QAPCritic(nn.Module):
    # MLP: 2→64→1, ~257 params
    # Input: mean-pooled ψ' → [B, 2]
    # Output: V(s) scalar → [B]
    # CRITICAL: psi_prime is DETACHED before critic head in ppo_agent.py
    #           to prevent value gradient from corrupting the shared encoder
```

---

## 6. PPO Training (Thesis §3.X.8)

### Current Hyperparameters (Phase 1b — v8)

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| PPO epochs per iteration | K | 3 | thesis spec |
| Clip epsilon | ε | 0.2 | thesis spec |
| Discount factor | γ | 0.99 | thesis spec |
| GAE lambda | λ_GAE | 0.95 | thesis spec |
| Value loss coefficient | c1 | 0.5 | thesis spec |
| Entropy bonus coefficient | c2 | **0.01** | thesis spec — 0.05 caused adv collapse |
| Learning rate | lr | 1e-4 | thesis spec |
| LR schedule | — | CosineAnnealingLR, eta_min=1e-5 | v4 fix: was 1e-6 |
| Batch size | B | **512** | Phase 1b: was 256 — larger batch → cleaner advantages |
| Epoch size | — | **128,000** | thesis spec — was 51,200 |
| Batches per epoch | — | **250** | 128,000 ÷ 512 |
| Total epochs | — | 200 | |
| Total opt steps | — | **1,200,000** | 200 × 250 × 3 × 8 |
| Gradient clipping | — | 1.0 | |
| kNN k | k | **10** | Phase 1: was 5 — covers 50% of N=20 |
| Lambda init | λ | 0.1 | |
| Rotation MLP hidden | — | 16 | |
| Inference augmentation | — | **×8 stochastic, best** | Phase 1: evaluate_augmented() |

### Why ENTROPY_COEF = 0.01 not 0.05

Phase 1 run (ENTROPY_COEF=0.05) showed:
- Entropy at ep200 = 0.546 — never collapsed, so 0.05 was not needed
- Clip fraction mean = 0.014% (healthy: 2–15%) — policy barely updating
- adv_std = 0.69 — advantage signal too weak
- 90% of learning done by epoch 50, then flat

Root cause: 0.05 coef kept entropy so high that sampled and greedy tours were nearly
identical, collapsing advantage estimates to near-zero. PPO ratios stayed at ~1.0,
clip fraction near-zero, no learning signal after ep50.

### PPO Loss

```
L = L_clip + c1 * L_value + c2 * L_entropy

L_clip = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
L_value = MSE(V(s_t), R_t)   [psi_prime DETACHED — critical bug P-enc]
L_entropy = -H[π(·|s_t)]

where r_t = π_new(a_t|s_t) / π_old(a_t|s_t)
      A_t = GAE advantages (normalized per rollout)
```

---

## 7. Scalability: K-Means Clustering (Thesis §3.X.2)

```python
from sklearn.cluster import KMeans

def cluster_instance(coords, demands, n_clusters):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(coords.numpy())
    clusters = []
    for k in range(n_clusters):
        mask = labels == k
        clusters.append({
            'coords': coords[mask],
            'demands': demands[mask],
            'indices': torch.where(torch.tensor(mask))[0]
        })
    return clusters
```

> After clustering, kNN must be **recomputed per sub-problem** (Pitfall P10).

---

## 8. Evaluation Protocol (Thesis §3.4)

### Three-Tier Evaluation

**Tier 1 — Solution Quality:**
Compare against OR-Tools, LKH-3 and AM, POMO, Sym-NCO.

**Tier 2 — Ablation Study (IMPLEMENTED):**
(a) full QAP-DRL  — `QAPPolicy(encoder_type="qap")`
(b) no QAP module — `QAPPolicy(encoder_type="baseline")` → `encoder/baseline_encoder.py`
(c) random perturbation, (d) no interference term — to be implemented
Run via: `python train_ablation_n20.py`

**Tier 3 — Generalization:**
Train on CVRP-50, test on CVRP-20 and CVRP-100 without fine-tuning.

### Internal Validation Method

Greedy rollout on the learned policy — NOT the classical nearest-neighbour heuristic.
At each step, `argmax` of the policy's output distribution (deterministic, no sampling).
Used during training because: (1) reflects what policy learned, (2) deterministic/low-variance,
(3) fast (~seconds vs 17min for OR-Tools), (4) standard in AM/POMO/DACT literature.

### Inference Augmentation (Phase 1)

`evaluate_augmented(model, instances, device, n_samples=8)`:
- Runs 8 stochastic rollouts per instance
- Takes element-wise minimum tour per instance across all 8 samples
- Returns mean of per-instance best tours
- Zero retraining cost — inference only
- Gain: ~2–4% gap reduction vs pure greedy

### Expected Performance Ranges

| Problem | LKH-3 | AM | POMO | QAP-DRL target |
|---------|-------|-----|------|----------------|
| CVRP-20 | ~6.10 | ~6.40 | ~6.20 | 6.15-6.35 |
| CVRP-50 | ~10.38 | ~10.98 | ~10.55 | 10.50-10.80 |
| CVRP-100 | ~15.65 | ~16.80 | ~15.90 | 15.80-16.50 |

### OR-Tools Reference Statistics

OR-Tools is run ONCE before training starts. Cached in `datasets/ortools_refs.json`.
The banner now prints:

```
OR-Tools Reference — CVRP-20  [cached]
Mean tour length  : 6.1915
Std deviation     : 0.8048  (CV = 13.0%)
Expected range    : 4.58 – 7.80  (mean ± 2σ)
Percentile distribution: p10 / p25 / p50 / p75 / p90
Valid / total     : 500 / 500  (0 failed)
Solve time/inst   : mean / max / n_time_limited
5% gap target     : ≤ 6.5011
Current best      : X.XXXX  (gap = +X.X%)
```

### Training Results Summary (CVRP-20)

| Run | ENTROPY_COEF | BATCH_SIZE | eta_min | Best tour | Gap | Notes |
|-----|-------------|-----------|---------|-----------|-----|-------|
| Run 5 | 0.02 | 256 | 1e-5 | 7.668 | 23.84% | LR floor fix confirmed working |
| Phase 1 run | 0.05 | 256 | 1e-5 | 7.228 | 17.15% | Entropy too high, adv collapse ep50 |
| Phase 1b (next) | **0.01** | **512** | 1e-5 | target: <15% | — | adv signal fix |

---

## 9. Methodological Note — Quantum Amplitude Contribution

The quantum amplitude encoding (ψ ∈ ℝ², ‖ψ‖=1) is mathematically equivalent to a
constrained linear projection (Linear + L2 norm). The rotation R(θ) is a standard 2D
orthogonal transformation.

**The thesis contribution requires empirical evidence via ablation** (Tier 2 above).
Three valid outcomes:
1. QAP-DRL gap < baseline gap → quantum structure contributes positively
2. QAP-DRL gap > baseline gap → important negative finding, publishable
3. Difference < 0.5% → null result, also publishable

Regardless of ablation outcome, the **parameter efficiency argument** stands independently:
~391 params vs ~1.4M in AM — a 3,500× reduction. This is a real and defensible contribution.

---

## 10. GPU / Device Handling

**Hardware:** NVIDIA GeForce RTX 3050, 4GB VRAM, CUDA 13.2
**Two machines:**
- Machine A: `C:\Users\ASUS\Downloads\Thesis Code QAP_VRP\cvrp-ppo\`
- Machine B: `D:\Coding\RubyThesis\QAP-CVRP\cvrp-ppo\`

**CRITICAL — Always run from inside cvrp-ppo/:**
```
cd D:\Coding\RubyThesis\QAP-CVRP\cvrp-ppo
python train_n20.py
```
Running from the parent directory causes `__file__` to resolve as relative, breaking dataset paths.

### VRAM Budget (RTX 3050)

| Problem | batch_size | Est. VRAM |
|---------|-----------|-----------|
| CVRP-20 | 512 | ~1.0 GB ✓ |
| CVRP-50 | 512 | ~2.0 GB ✓ |
| CVRP-100 | 256 | ~2.0 GB ✓ |
| CVRP-100 | 512 | ~4.0 GB ⚠ |
