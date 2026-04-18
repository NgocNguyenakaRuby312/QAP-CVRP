# KNOWLEDGE: QAP-DRL Architecture & Methodology

**Purpose:** Complete mathematical specification and PyTorch implementation reference for every QAP-DRL component.
**When to reference:** Any coding, implementation, or architecture question. This is the single source of truth.
**Last updated:** March 2026

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
# Validation: 10,000 fixed instances per problem size
# Test: 10,000 fixed instances per problem size
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
    """
    Thesis §3.X.3: Build 5D feature vector per node.

    Args:
        coords: [B, N+1, 2]  — (x, y) for depot + N customers
        demands: [B, N+1]    — demand (depot demand = 0)
        capacity: float      — vehicle capacity C
    Returns:
        features: [B, N+1, 5]

    Feature order (per thesis §3.X.3):
        [0] d_i/C          — demand-to-capacity ratio
        [1] dist(i, depot) — Euclidean distance to depot
        [2] x_i            — x coordinate
        [3] y_i            — y coordinate
        [4] α_i/π          — normalized angular position (atan2 / π)
    """
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

Maps 5D features to unit circle in 2D.

```
ψ_i = Normalize(W_proj · x_i + b_proj)
where W_proj ∈ ℝ^{2×5}, b_proj ∈ ℝ^2
s.t. α_i² + β_i² = 1
```

```python
class AmplitudeProjection(nn.Module):
    def __init__(self, input_dim=5, amp_dim=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, amp_dim)  # W ∈ R^{2×5}, b ∈ R^2

    def forward(self, x):
        """
        Args:
            x: [B, N+1, 5]
        Returns:
            psi: [B, N+1, 2] — unit norm vectors (α² + β² = 1)
        """
        projected = self.proj(x)                           # [B, N+1, 2]
        psi = F.normalize(projected, p=2, dim=-1)          # [B, N+1, 2] norm=1
        return psi
```

**Critical:** L2 normalization is NON-NEGOTIABLE. Every ψ_i must have ||ψ_i|| = 1.

### 3.3 Rotation MLP (Thesis §3.X.5)

Maps 5D features to a per-node rotation angle θ_i.

**Thesis §3.X.5 specifies:** single hidden layer of **16 units** and **hyperbolic tangent** activation.

```
θ_i = MLP(x_i)   where MLP: ℝ^5 → ℝ^16 → ℝ^1
```

```python
class RotationMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 5×16 + 16 = 96
            nn.Tanh(),                           # thesis specifies tanh
            nn.Linear(hidden_dim, 1)             # 16×1 + 1 = 17
        )                                        # Total: ~113 params

    def forward(self, x):
        """
        Args:
            x: [B, N+1, 5]
        Returns:
            theta: [B, N+1] — rotation angles in radians
        """
        theta = self.mlp(x).squeeze(-1)                    # [B, N+1]
        return theta
```

### 3.4 Rotation Matrix Application (Thesis §3.X.5)

```
R(θ) = [[cos θ, -sin θ],
        [sin θ,  cos θ]]
ψ'_i = R(θ_i) · ψ_i
```

```python
def apply_rotation(psi, theta):
    """
    Args:
        psi: [B, N+1, 2]    — amplitude vectors (unit norm)
        theta: [B, N+1]     — rotation angles
    Returns:
        psi_prime: [B, N+1, 2] — rotated amplitudes (still unit norm)
    """
    cos_t = torch.cos(theta)                               # [B, N+1]
    sin_t = torch.sin(theta)                               # [B, N+1]
    alpha = psi[:, :, 0]                                   # [B, N+1]
    beta  = psi[:, :, 1]                                   # [B, N+1]
    alpha_prime = cos_t * alpha - sin_t * beta             # [B, N+1]
    beta_prime  = sin_t * alpha + cos_t * beta             # [B, N+1]
    psi_prime = torch.stack([alpha_prime, beta_prime], dim=-1)  # [B, N+1, 2]
    return psi_prime
    # No re-normalization needed — rotation preserves norm
```

### 3.5 Complete Encoder

```python
class QAPEncoder(nn.Module):
    def __init__(self, input_dim=5, amp_dim=2, hidden_dim=16):
        super().__init__()
        self.amplitude_proj = AmplitudeProjection(input_dim, amp_dim)
        self.rotation_mlp = RotationMLP(input_dim, hidden_dim)

    def forward(self, features):
        """
        Args:
            features: [B, N+1, 5]
        Returns:
            psi_prime: [B, N+1, 2] — rotated amplitude embeddings (unit norm)
        """
        psi = self.amplitude_proj(features)                # [B, N+1, 2]
        theta = self.rotation_mlp(features)                # [B, N+1]
        psi_prime = apply_rotation(psi, theta)             # [B, N+1, 2]
        return psi_prime
```

---

## 4. Decoder Components (Thesis §3.X.6–3.X.7)

### 4.1 Context Query Construction (Thesis §3.X.6)

```
context_t = [ψ'_{curr}, remaining_cap / C, t / N]  ∈ ℝ^4
q_t = W_q · context_t                                ∈ ℝ^2
where W_q ∈ ℝ^{2×4}
```

**ψ′_curr = zero vector if at the depot.**

```python
class ContextQueryLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(4, 2, bias=False)   # 8 params, no bias

    def forward(self, psi_curr, cap_remaining, capacity, step, N):
        """
        Args:
            psi_curr: [B, 2]      — amplitude of current node (zero if at depot)
            cap_remaining: [B]    — remaining vehicle capacity
            capacity: float       — max capacity C
            step: int             — current step
            N: int                — total customers
        Returns:
            query: [B, 2]
        """
        cap_ratio = (cap_remaining / capacity).unsqueeze(-1)   # [B, 1]
        step_ratio = torch.full_like(cap_ratio, step / N)      # [B, 1]
        ctx = torch.cat([psi_curr, cap_ratio, step_ratio], dim=-1)  # [B, 4]
        query = self.W_q(ctx)                                  # [B, 2]
        return query
```

### 4.2 Hybrid Scoring Mechanism (Thesis §3.X.7)

```
S_context(j) = q_t · ψ'_j
E_kNN(j) = Σ_{i ∈ kNN(j)} (ψ'_i · ψ'_j)
Score(j) = S_context(j) + λ · E_kNN(j)
where λ is a learnable scalar (initialized at 0.1)
```

```python
class HybridScoringLayer(nn.Module):
    def __init__(self, lambda_init=0.1):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, query, psi_prime, knn_indices, feasible_mask, top_m=None):
        """
        Args:
            query: [B, 2]
            psi_prime: [B, N+1, 2]
            knn_indices: [B, N+1, k]
            feasible_mask: [B, N+1]    — True = INFEASIBLE
            top_m: Optional[int]
        Returns:
            log_probs: [B, N+1]
        """
        # Term 1 — Context attention
        context_scores = torch.bmm(
            psi_prime, query.unsqueeze(-1)
        ).squeeze(-1)                                      # [B, N+1]

        # Term 2 — Interference
        B, N1, k = knn_indices.shape
        idx = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 2)  # [B, N+1, k, 2]
        neighbor_psi = torch.gather(
            psi_prime.unsqueeze(2).expand(-1, -1, k, -1), 1, idx
        )                                                  # [B, N+1, k, 2]
        psi_j = psi_prime.unsqueeze(2)                     # [B, N+1, 1, 2]
        dots = (psi_j * neighbor_psi).sum(dim=-1)          # [B, N+1, k]
        interference = dots.sum(dim=-1)                    # [B, N+1]

        # Combined score
        scores = context_scores + self.lambda_param * interference  # [B, N+1]

        # Mask infeasible BEFORE softmax
        scores[feasible_mask] = -1e9

        if top_m is not None and top_m < N1:
            _, top_indices = scores.topk(top_m, dim=-1)
            top_mask = torch.ones_like(scores, dtype=torch.bool)
            top_mask.scatter_(1, top_indices, False)
            scores[top_mask & ~feasible_mask] = -1e9

        log_probs = F.log_softmax(scores, dim=-1)          # [B, N+1]
        return log_probs
```

### 4.3 kNN Precomputation

```python
def compute_knn(coords, k=5):
    """
    Args:
        coords: [B, N+1, 2]
    Returns:
        knn_indices: [B, N+1, k]
    """
    dists = torch.cdist(coords, coords)                    # [B, N+1, N+1]
    dists.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))   # no self-loops
    _, knn_indices = dists.topk(k, dim=-1, largest=False)  # [B, N+1, k]
    return knn_indices
```

### 4.4 Feasibility Masking

```python
def get_feasibility_mask(visited, demands, remaining_cap):
    """
    Args:
        visited: [B, N+1]       — boolean, True = already visited
        demands: [B, N+1]
        remaining_cap: [B]
    Returns:
        mask: [B, N+1]          — True = INFEASIBLE
    """
    mask = visited.clone()
    exceeds_cap = demands > remaining_cap.unsqueeze(-1)
    mask = mask | exceeds_cap
    mask[:, 0] = False   # depot ALWAYS feasible
    return mask
```

### 4.5 Autoregressive Decoding Loop

```python
def decode(encoder_output, env, context_query, scoring, knn_indices, greedy=False):
    state = env.reset()
    psi_prime = encoder_output                              # [B, N+1, 2]
    log_probs_collected = []
    actions_collected = []

    while not state.all_done:
        psi_curr = psi_prime[range(B), state.current_node]  # [B, 2]
        at_depot = (state.current_node == 0)
        psi_curr[at_depot] = 0.0                            # zero vector at depot

        query = context_query(psi_curr, state.remaining_cap,
                              state.capacity, state.step, N)  # [B, 2]

        mask = get_feasibility_mask(state.visited, state.demands, state.remaining_cap)
        log_probs = scoring(query, psi_prime, knn_indices, mask)  # [B, N+1]

        if greedy:
            action = log_probs.argmax(dim=-1)               # [B]
        else:
            dist = Categorical(logits=log_probs)
            action = dist.sample()                           # [B]

        log_probs_collected.append(dist.log_prob(action) if not greedy else None)
        actions_collected.append(action)
        state = env.step(action)

    return actions_collected, log_probs_collected, state
```

---

## 5. Critic Network

```python
class QAPCritic(nn.Module):
    def __init__(self, amp_dim=2, hidden_dim=64):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(amp_dim, hidden_dim),       # 2×64 + 64 = 192
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)              # 64×1 + 1 = 65
        )                                          # Total: ~257 params

    def forward(self, psi_prime):
        """
        Args:
            psi_prime: [B, N+1, 2]
        Returns:
            value: [B]
        """
        pooled = psi_prime.mean(dim=1)                     # [B, 2]
        value = self.value_head(pooled).squeeze(-1)        # [B]
        return value
```

---

## 6. PPO Training (Thesis §3.X.8)

### Hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| PPO epochs per iteration | K | 3 |
| Clip epsilon | ε | 0.2 |
| Discount factor | γ | 0.99 |
| GAE lambda | λ_GAE | 0.95 |
| Value loss coefficient | c1 | 0.5 |
| Entropy bonus coefficient | c2 | 0.01 |
| Learning rate | lr | 1e-4 |
| Batch size | B | 256 (RTX 3050 default) |
| Epoch size | — | 128,000 |
| Total epochs | — | 100-200 |
| Gradient clipping | — | 1.0 |
| kNN k | k | 5 |
| Lambda init | λ | 0.1 |
| Rotation MLP hidden | — | 16 |

### PPO Loss

```
L = L_clip + c1 * L_value + c2 * L_entropy

L_clip = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
L_value = MSE(V(s_t), R_t)
L_entropy = -H[π(·|s_t)]

where r_t = π_new(a_t|s_t) / π_old(a_t|s_t)
      A_t = GAE advantages
```

### Sensitivity Analysis Parameters (Thesis §3.5)

Perturbation strength α ∈ {0.01, 0.05, 0.10, 0.20, 0.50}

---

## 7. Scalability: K-Means Clustering (Thesis §3.X.2)

```python
from sklearn.cluster import KMeans

def cluster_instance(coords, demands, n_clusters):
    """
    Args:
        coords: [N, 2]       — customer coordinates (no depot)
        demands: [N]
        n_clusters: int
    Returns:
        clusters: list of dicts with keys: coords, demands, indices
    """
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
Compare against Gurobi, LKH-3, OR-Tools and AM, POMO, Sym-NCO.

**Tier 2 — Ablation Study:**
(a) full model, (b) no QAP module, (c) no MHA encoder, (d) random perturbation.

**Tier 3 — Generalization:**
Train on CVRP-50, test on CVRP-20 and CVRP-100 without fine-tuning.

### Expected Performance Ranges

| Problem | LKH-3 | AM | POMO | QAP-DRL target |
|---------|-------|-----|------|----------------|
| CVRP-20 | ~6.10 | ~6.40 | ~6.20 | 6.15-6.35 |
| CVRP-50 | ~10.38 | ~10.98 | ~10.55 | 10.50-10.80 |
| CVRP-100 | ~15.65 | ~16.80 | ~15.90 | 15.80-16.50 |

---

## 9. GPU / Device Handling

**Hardware:** NVIDIA GeForce RTX 3050, 4GB VRAM, CUDA 13.2
**PyTorch build:** cu121 (`pip install torch --index-url https://download.pytorch.org/whl/cu121`)

### Device Detection (run.py only — pass everywhere else as argument)
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### All Modules Must Accept `device` as Argument
```python
# data_generator.py
def generate_instance(B, N, capacity, device):
    coords  = torch.FloatTensor(B, N+1, 2).uniform_(0, 1).to(device)
    demands = torch.zeros(B, N+1, dtype=torch.long).to(device)
    demands[:, 1:] = torch.randint(1, 10, (B, N), device=device)
    return coords, demands, capacity

# knn.py
def compute_knn(coords, k=5):
    # coords already on device
    dists = torch.cdist(coords, coords)          # stays on device
    dists.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    _, knn_indices = dists.topk(k, dim=-1, largest=False)
    return knn_indices                           # [B, N+1, k] on device
```

### VRAM Budget (RTX 3050)

| Problem | batch_size | Est. VRAM |
|---------|-----------|-----------|
| CVRP-20 | 256 | ~0.5 GB ✓ |
| CVRP-50 | 256 | ~1.0 GB ✓ |
| CVRP-100 | 256 | ~2.0 GB ✓ |
| CVRP-100 | 512 | ~4.0 GB ⚠ |

**Default batch_size = 256.**

### Memory Management
```python
# Start of every epoch in ppo_agent.py
torch.cuda.empty_cache()
```
