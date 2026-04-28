# KNOWLEDGE: Glossary & Math-to-Code Mapping

**Purpose:** Quick-reference mapping thesis notation to code variables, terminology definitions, and error decoder.
**When to reference:** Naming variables, interpreting thesis equations, understanding error messages.
**Last updated:** March 2026

---

## 1. Thesis Notation → Code Variables

### Feature Construction (§3.X.3)

| Thesis | Code | Shape | Description |
|--------|------|-------|-------------|
| `xᵢ` | `features` | `[B, N+1, 5]` | 5D feature vector per node |
| `dᵢ/C` | `features[:,:,0]` | `[B, N+1]` | Demand-to-capacity ratio |
| `dist(i, depot)` | `features[:,:,1]` | `[B, N+1]` | Euclidean distance to depot |
| `xᵢ` (coord) | `features[:,:,2]` | `[B, N+1]` | x coordinate |
| `yᵢ` (coord) | `features[:,:,3]` | `[B, N+1]` | y coordinate |
| `αᵢ/π` | `features[:,:,4]` | `[B, N+1]` | Normalized angular position |
| `C` | `capacity` | scalar | Vehicle capacity |
| `N` | `graph_size` | scalar | Number of customers (depot excluded) |

### Amplitude Projection (§3.X.4)

| Thesis | Code | Shape | Description |
|--------|------|-------|-------------|
| `W` | `self.proj.weight` | `(2, 5)` | Projection matrix |
| `b` | `self.proj.bias` | `(2,)` | Projection bias |
| `[αᵢ, βᵢ]ᵀ` | `psi` | `[B, N+1, 2]` | Amplitude vector (unit norm) |
| `\|ψᵢ⟩` | `psi` | `[B, N+1, 2]` | Quantum-inspired state notation |
| `Normalize(·)` | `F.normalize(·, p=2, dim=-1)` | — | L2 normalization to unit circle |

### Rotation (§3.X.5)

| Thesis | Code | Shape | Description |
|--------|------|-------|-------------|
| `θᵢ` | `theta` | `[B, N+1]` | Per-node rotation angle |
| `MLP(xᵢ)` | `self.rotation_mlp(features)` | `[B, N+1, 1]` → squeeze | MLP: 5→16→1, tanh |
| `R(θᵢ)` | inline `cos_t`, `sin_t` | `(2,2)` per node | 2D rotation matrix |
| `\|ψ'ᵢ⟩` | `psi_prime` | `[B, N+1, 2]` | Rotated amplitude (still unit norm) |
| `cos θᵢ` | `cos_t` | `[B, N+1]` | Cosine component |
| `sin θᵢ` | `sin_t` | `[B, N+1]` | Sine component |

### Context & Query (§3.X.6)

| Thesis | Code | Shape | Description |
|--------|------|-------|-------------|
| `contextₜ` | `ctx` | `[B, 4]` | [ψ'_curr(2), cap/C(1), t/N(1)] |
| `ψ'_curr` | `psi_curr` | `[B, 2]` | Current node amplitude (zero if at depot) |
| `remaining_cap/C` | `cap_ratio` | `[B, 1]` | Capacity fraction remaining |
| `t/N` | `step_ratio` | `[B, 1]` | Decoding progress |
| `W_q` | `self.W_q` | `nn.Linear(4,2,bias=False)` | Query projection (8 params) |
| `queryₜ` | `query` | `[B, 2]` | Query in amplitude space |

### Scoring (§3.X.7)

| Thesis | Code | Shape | Description |
|--------|------|-------|-------------|
| `Score(j)` | `scores` | `[B, N+1]` | Combined score per node |
| `S_context(j) = q·ψ'_j` | `context_scores` | `[B, N+1]` | Attention term |
| `E_kNN(j) = Σ ψ'_i·ψ'_j` | `interference` | `[B, N+1]` | Interference term |
| `λ` | `self.lambda_param` | `nn.Parameter(scalar)` | Learnable balance (init 0.1) |
| `kNN(j)` | `knn_indices` | `[B, N+1, k]` | k-nearest neighbor indices |
| `P(j)` | `probs` or `log_probs` | `[B, N+1]` | Action probabilities |
| `k` | `knn_k` | scalar (default 5) | Number of neighbors |

### PPO (§3.X.8)

| Thesis | Code | Shape | Description |
|--------|------|-------|-------------|
| `R` | `reward` | `[B]` | −Total Distance |
| `V(sₜ)` | `value` | `[B]` | Critic output |
| `Aₜ` | `advantages` | `[B, T]` | GAE advantages |
| `rₜ(θ)` | `ratio` | `[B, T]` | π_new / π_old |
| `ε` | `eps_clip` | 0.2 | PPO clip threshold |
| `L_clip` | `clip_loss` | scalar | Clipped surrogate loss |
| `L_value` | `value_loss` | scalar | MSE(V, returns) |
| `L_entropy` | `entropy_loss` | scalar | −H[π] |
| `c1` | `c1` | 0.5 | Value loss weight |
| `c2` | `c2` | 0.01 | Entropy bonus weight |
| `γ` | `gamma` | 0.99 | Discount factor |
| `λ_GAE` | `gae_lambda` | 0.95 | GAE lambda |

---

## 2. Terminology

### Quantum-Inspired (Classical, NOT Quantum Computing)

| Term | In This Project | NOT This |
|------|----------------|----------|
| Amplitude | 2D vector [α, β] on unit circle | Complex quantum amplitudes |
| Unit norm | α² + β² = 1 (L2 normalization) | Born rule |
| Rotation | 2D rotation matrix R(θ), orthogonal | Quantum gate (RY, RZ) |
| Interference | Dot-product similarity Σψ'_i·ψ'_j | Quantum wave interference |
| State \|ψ⟩ | Classical 2D vector, Dirac notation is analogy | Quantum superposition |
| Unitary | Orthogonal rotation (preserves norm) | Unitary quantum operator |
| Perturbation | Learned rotation of amplitude vector | Quantum measurement perturbation |
| Constructive / destructive | High/low dot-product similarity | Quantum interference patterns |

### Vehicle Routing

| Term | Definition |
|------|-----------|
| **CVRP** | Capacitated Vehicle Routing Problem |
| **Depot** | Start/end point (node index 0) |
| **Tour / Route** | Sequence by one vehicle: depot → customers → depot |
| **Feasible** | All constraints satisfied (capacity, all-visited, depot start/end) |
| **Optimality gap** | (cost − best_known) / best_known × 100% |
| **K-Means decomposition** | Splitting N nodes into K spatial clusters (§3.X.2) |
| **Autoregressive** | Build solution one node at a time (constructive) |
| **Constructive** | Build from scratch (vs improvement-based like DACT's 2-opt) |

### RL / PPO

| Term | Definition |
|------|-----------|
| **PPO** | Proximal Policy Optimization (§3.X.8) |
| **Actor** | Encoder + decoder (Stages 1-6) — the route construction policy |
| **Critic** | MLP estimating V(s) for advantage computation |
| **GAE** | Generalized Advantage Estimation (λ_GAE=0.95) |
| **Advantage** | A_t = GAE-estimated return − V(s), how much better than expected |
| **Clipping** | Constraining policy ratio to [1-ε, 1+ε] for stable updates |
| **Greedy decoding** | Always pick highest-probability node (inference) |
| **Sampling** | Sample from Categorical distribution (training) |
| **Rollout** | Complete episode of route construction |

---

## 3. Error Message Decoder

### PyTorch Errors

| Error Pattern | Meaning | Fix |
|--------------|---------|-----|
| `element 0 does not require grad` | Gradient chain broken | Find `torch.tensor()`, `.detach()`, or `.item()` in computation path |
| `modified by an inplace operation` | In-place op in autograd graph | Replace `x[:,0] = val` with `torch.cat()` or `scatter_` |
| `CUDA out of memory` | Batch × nodes too large | Reduce batch_size, use gradient accumulation |
| `Expected all tensors on same device` | CPU/GPU mismatch | Add `.to(device)` |
| `mat1 and mat2 shapes cannot be multiplied` | Linear layer dim wrong | Check `nn.Linear(in_features)` matches input last dim |
| `index out of range in self` | Bad index in embedding/gather | Check max index < tensor size on that dim |
| `Categorical invalid probs` | Negative or NaN probabilities | Check masking produces valid distribution; use log_softmax |

### QAP-DRL Specific Symptoms

| Symptom | Likely Cause | First Check |
|---------|-------------|-------------|
| Norm ≠ 1 after projection | Missing `F.normalize` | Pitfall P1 |
| Norm ≠ 1 after rotation | Rotation matrix wrong | Check cos/sin placement |
| All scores identical | λ=0 or query=zero | Print both terms separately |
| kNN gives wrong neighbors | Computed in feature space | Use coords, not features (Pitfall P11) |
| λ not updating | Defined as float | Use `nn.Parameter(torch.tensor(...))` |
| θ all same | MLP collapsed | Check init, verify feature variation |
| Infeasible routes | Mask bug | Print mask each step (Pitfall P2, P3) |
| PPO ratio >> 1 | Old log-probs stale | Store before update, detach |
| Reward flat | Wrong sign or LR | R = −distance; try LR × 10 |
| Entropy = 0 | Policy collapsed | Increase c2; verify sampling not greedy |
| Angle feature > 1.0 | Missing /π | Pitfall P12 |
| Feature order wrong | [x,y,...] not [d/C,...] | Check against §3.X.3 (Pitfall P11) |
