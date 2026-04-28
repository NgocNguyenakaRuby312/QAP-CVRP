# KNOWLEDGE: Debugging Guide & Common Pitfalls

**Purpose:** Quick-reference for diagnosing and fixing common issues in QAP-DRL implementation.
**When to reference:** Any error, unexpected behavior, NaN/Inf, shape mismatch, or convergence problem.
**Last updated:** April 2026 (Phase 2: 4D amplitudes)

---

## 1. Shape Mismatch Cheat Sheet

```
INPUT
  coords:           [B, N+1, 2]
  demands:          [B, N+1]        (depot demand = 0)
  capacity:         scalar float

FEATURE CONSTRUCTION
  features:         [B, N+1, 5]     order: [d/C, dist_depot, x, y, angle/pi]

ENCODER — QAP mode (Phase 2: 4D)
  psi (after proj):     [B, N+1, 4]     unit norm on S³!
  theta (from MLP):     [B, N+1, 6]     6 Givens angles for SO(4)
  psi_prime (rotated):  [B, N+1, 4]     still unit norm on S³!

ENCODER — baseline mode (ablation)
  embedding:            [B, N+1, 4]     NOT unit norm (ReLU output)

KNN
  knn_indices:      [B, N+1, k]     k=10(N=20), k=15(N=50), k=20(N=100), k=30(N=200)

DECODER (per step t)
  psi_curr:         [B, 4]          current node amplitude (zero if at depot)
  context:          [B, 8]          [psi_curr(4), cap_ratio(1), step_ratio(1), x_curr(1), y_curr(1)]
  query:            [B, 4]          W_q * context  (W_q is 4×8, no bias)
  current_coords:   [B, 2]          vehicle Euclidean position (returned by context_query)
  context_scores:   [B, N+1]        query · psi_j
  interference:     [B, N+1]        sum_kNN psi_i · psi_j
  dist_to_nodes:    [B, N+1]        norm(coords_j - current_coords)
  scores:           [B, N+1]        context_scores + lambda*interference - mu*dist_to_nodes
  log_probs:        [B, N+1]        log_softmax(masked scores)
  action:           [B]             selected node index

CRITIC (psi_prime DETACHED)
  pooled:           [B, 4]          mean pool of psi_prime
  value:            [B]             MLP(4→64→1)

PPO
  advantages:       [B, T]
  returns:          [B, T]
  old_log_probs:    [B, T]

evaluate_actions() PPO update path
  ctx:              [mb, T, 8]      (D+4=8)
  query:            [mb, T, 4]
  cur_coords_3d:    [mb, T, 2]
  psi_prime_3d:     [mb, T, N+1, 4] BROADCAST from static psi_prime
  dist_to_nodes:    [mb, T, N+1]
  scores:           [mb, T, N+1]
```

---

## 2. Known Pitfalls

| # | Pitfall | Fix |
|---|---------|-----|
| P1 | Missing L2 norm after projection | `F.normalize(p=2, dim=-1)` — unit norm on S³ |
| P2 | Mask after softmax | -1e9 BEFORE log_softmax |
| P3 | Depot masked | `mask[:, 0] = False` |
| P4 | kNN self-loops | `diagonal.fill_(inf)` before topk |
| P5 | NaN in rotation | `torch.clamp(theta, -10, 10)` |
| P6 | Episode length mismatch | Track carefully: N selections + depot returns |
| P7 | Capacity not reset on depot | `env.step()`: action==0 → remaining_cap = capacity |
| P8 | Depot demand not zero | `demands[:, 0] = 0` |
| P9 | Gradient through sampling | Use `Categorical`, collect `log_prob(action)` |
| P10 | kNN stale after clustering | Recompute kNN per sub-problem |
| P11 | Feature order wrong | `[d/C, dist_depot, x, y, angle/pi]` — 5D |
| P12 | Angle not normalized by π | Divide atan2 by `math.pi` |
| P13 | CUDA/CPU device mismatch | `.to(device)` everywhere |
| P14 | OOM on RTX 3050 | Reduce batch_size; `torch.cuda.empty_cache()` |
| P15 | context_query returns tuple | Unpack: `query, current_coords = context_query(...)` |
| P16 | mu_param not nn.Parameter | `nn.Parameter(torch.tensor(0.5))` |
| P17 | feature_dim set to 6 | Must be 5. Encoder is static 5D |
| P18 | Per-step re-encoding | NEVER. psi_prime is fixed after initial encode |
| P19 | evaluate_actions rebuilds psi_prime | Must BROADCAST static psi_prime |
| P20 | Post-training reload missing amp_dim/hidden_dim | `QAPPolicy(..., amp_dim=AMP_DIM, hidden_dim=HIDDEN_DIM)` |
| P21 | Optimizer state dict mismatch on resume | Delete old epoch files when changing optimizer structure |
| P24 | Dense grey gridlines on twinx | `.set_zorder(-1)`, `.patch.set_visible(False)`, `.grid(False)` |

---

## 3. Training Convergence Diagnostics

### Clip Fraction
| clip_fraction | Interpretation | Action |
|---|---|---|
| < 0.1% | Policy not updating | Reduce entropy_coef; increase batch_size |
| 2% - 15% | Healthy | No action |
| > 30% | Over-clipping | Reduce LR |

### adv_std
| adv_std | Interpretation |
|---|---|
| > 1.5 | Strong signal (early) |
| 0.8 - 1.5 | Healthy |
| < 0.5 | Near-collapse |

### mu Behaviour (Phase 2)
- mu at 0: model relies on amplitude geometry alone
- mu 1-10: healthy balance of distance + amplitude
- mu at ceiling (20): distance penalty dominant — model may be nearest-neighbour-ish
- With 4D amplitudes, higher mu is acceptable since amplitude space provides complementary signal

### lambda Behaviour (Phase 2)
- λ positive: interference amplifies similar neighbours
- λ near zero: interference not contributing
- λ negative: interference used as diversity penalty (common in 4D — model wants to distinguish nodes)
- λ at floor (-2): interference strongly penalized — may need to lower floor further

---

## 4. Chart Rendering Fixes

All twinx() axes:
```python
ax_twin.set_zorder(primary.get_zorder() - 1)
ax_twin.patch.set_visible(False)
ax_twin.grid(False)
```

Tick label overflow:
```python
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))
```

---

## 5. Path / Launch Error

Always run from inside `cvrp-ppo/`:
```
cd "C:\Users\ASUS\Downloads\Thesis Code QAP_VRP\cvrp-ppo"
python train_n20.py
```

---

## 6. Validation Checklist

```python
# Encoder (QAP mode — 4D)
assert features.shape == (B, N+1, 5),     "5D features"
assert psi_prime.shape == (B, N+1, 4),     "4D amplitudes on S³"
norms = psi_prime.norm(dim=-1)
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "unit norm"

# Context (Phase 2)
assert ctx.shape[-1] == 8,               "context must be 8D (D+4)"
assert current_coords.shape == (B, 2)
assert query.shape == (B, 4),            "query must be 4D"

# Scoring
assert hasattr(model.decoder.hybrid, 'mu_param')
assert dist_to_nodes.shape == (B, N+1)

# Masking
assert mask[:, 0].sum() == 0,            "depot never masked"

# Policy creation consistency
# ALL QAPPolicy() calls must include amp_dim=AMP_DIM, hidden_dim=HIDDEN_DIM
```

---

## 7. Debugging Workflow

1. Identify component: encoder / decoder / environment / PPO / data / device / chart
2. Check P1-P21 pitfalls above
3. Verify shapes against Section 1 cheat sheet
4. Check clip_fraction in train_log.jsonl
5. Minimal reproduction: B=2, N=5
6. Check launch directory — must be inside cvrp-ppo/
