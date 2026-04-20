# KNOWLEDGE: Debugging Guide & Common Pitfalls

**Purpose:** Quick-reference for diagnosing and fixing common issues in QAP-DRL implementation.
**When to reference:** Any error, unexpected behavior, NaN/Inf, shape mismatch, or convergence problem.
**Last updated:** April 2026

---

## 1. Shape Mismatch Cheat Sheet

Expected tensor shapes at every key point in the forward pass:

```
INPUT
  coords:           [B, N+1, 2]
  demands:          [B, N+1]        (depot demand = 0)
  capacity:         scalar float

FEATURE CONSTRUCTION (§3.X.3)
  features:         [B, N+1, 5]     order: [d/C, dist, x, y, angle/π]

ENCODER (§3.X.4-5) — QAP mode
  psi (after proj):     [B, N+1, 2]     ← unit norm!
  theta (from MLP):     [B, N+1]
  psi_prime (rotated):  [B, N+1, 2]     ← still unit norm!

ENCODER — baseline mode (ablation)
  embedding:            [B, N+1, 2]     ← NOT unit norm (ReLU output)

KNN
  knn_indices:      [B, N+1, k]     k=10 for CVRP-20, k=5 for CVRP-50/100

DECODER (per step t, §3.X.6-7)
  psi_curr:         [B, 2]          ← current node's amplitude (zero if at depot)
  context:          [B, 4]          ← [ψ'_curr(2), cap_ratio(1), step_ratio(1)]
  query:            [B, 2]          ← W_q · context
  context_scores:   [B, N+1]        ← query · ψ'_j
  interference:     [B, N+1]        ← Σ_kNN ψ'_i · ψ'_j
  scores:           [B, N+1]        ← context_scores + λ · interference
  log_probs:        [B, N+1]        ← log_softmax(masked scores)
  action:           [B]             ← selected node index

CRITIC
  pooled:           [B, 2]          ← mean pool of psi_prime (DETACHED)
  value:            [B]             ← MLP(pooled)

PPO
  advantages:       [B, T]          where T = episode length
  returns:          [B, T]
  old_log_probs:    [B, T]
```

---

## 2. Top 14 Known Pitfalls

### P1: Forgetting L2 Normalization After Projection
**Symptom:** Amplitude vectors have varying norms; rotation changes magnitude; scores explode.
**Fix:** Always apply `F.normalize(projected, p=2, dim=-1)` immediately after `self.proj(x)`.
**Verify:** `assert torch.allclose(psi.norm(dim=-1), torch.ones_like(psi[:,:,0]), atol=1e-5)`
**Note:** Only applies in QAP mode. Baseline encoder (`encoder_type="baseline"`) intentionally has no L2 norm.

### P2: Masking AFTER Softmax Instead of BEFORE
**Symptom:** Infeasible nodes get non-zero probability; invalid tours.
**Fix:** Set `scores[mask] = -1e9` BEFORE `F.log_softmax(scores, dim=-1)`.
**Never:** Apply mask after softmax or use multiplicative masking.
**Note:** Use `-1e9` not `float('-inf')` — inf can produce NaN if all nodes masked.

### P3: Masking the Depot
**Symptom:** Vehicles can't return to depot; episode never terminates; infinite loops.
**Fix:** `mask[:, 0] = False` — depot is ALWAYS feasible regardless of visited/capacity state.

### P4: Self-Loops in kNN
**Symptom:** Interference term includes node's own amplitude; scores biased upward.
**Fix:** Set diagonal to infinity before topk: `dists.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))`.

### P5: NaN in Rotation When theta is NaN/Inf
**Symptom:** NaN propagates through cos/sin into psi_prime and beyond.
**Fix:** Clamp theta output: `theta = torch.clamp(theta, -10.0, 10.0)`.
**Root cause:** Often from exploding gradients in RotationMLP. Check `max_grad_norm` clipping.

### P6: Episode Length Mismatch
**Symptom:** PPO buffer has inconsistent lengths; indexing errors in advantage calculation.
**Fix:** Episode length = exactly N customer selections + depot returns. Track carefully.

### P7: Capacity Not Reset on Depot Return
**Symptom:** After returning to depot, vehicle still has reduced capacity; customers become infeasible.
**Fix:** In `env.step()`, when action == 0 (depot): `remaining_cap = capacity`.

### P8: Demand of Depot Not Zero
**Symptom:** Visiting depot reduces capacity; cascading feasibility issues.
**Fix:** Ensure `demands[:, 0] = 0` during data generation and never modify it.

### P9: Gradient Through Sampling
**Symptom:** Gradients are None or zero for the policy.
**Fix:** Use `torch.distributions.Categorical` for sampling, collect `log_prob(action)` for PPO.

### P10: kNN Indices Stale After Clustering
**Symptom:** kNN references nodes from other clusters; index out of bounds.
**Fix:** Recompute kNN per sub-problem after clustering.

### P11: Feature Order Mismatch
**Symptom:** Model trains but performs poorly; amplitude projection learns wrong feature mapping.
**Fix:** Verify feature order matches thesis §3.X.3: `[d/C, dist, x, y, angle/π]`.

### P12: Angle Not Normalized by π
**Symptom:** Feature index [4] has range [-π, π] instead of [-1, 1]; dominates other features.
**Fix:** Divide atan2 output by `math.pi`.

### P13: Tensor on Wrong Device (CUDA/CPU Mismatch)
**Symptom:** `RuntimeError: Expected all tensors to be on the same device`
**Fix:** `.to(device)` on all tensors at generation time. Never hardcode "cuda"/"cpu".

### P14: CUDA Out of Memory (OOM) on RTX 3050
**Symptom:** `torch.cuda.OutOfMemoryError`
**Fix sequence:**
```python
batch_size = 256    # reduce from 512 if OOM (CVRP-50/100 with B=512)
torch.cuda.empty_cache()
with torch.no_grad():
    results = evaluate(model, val_data)
```

---

## 3. Training Convergence Diagnostics

### Clip Fraction is the Key Indicator

| clip_fraction | Interpretation | Action |
|--------------|----------------|--------|
| < 0.1% | Policy not updating — advantage signal too weak | Reduce entropy_coef; increase batch_size |
| 0.1% – 2% | Weak updates — borderline | Monitor adv_std |
| 2% – 15% | **Healthy range** | No action needed |
| 15% – 30% | Policy changing fast | Reduce LR or increase ppo_epochs |
| > 30% | Over-clipping — policy unstable | Reduce LR drastically |

**Phase 1 run had clip_fraction = 0.014% (mean) — critically low.**
Root cause: ENTROPY_COEF=0.05 kept entropy at H=0.55, making sampled ≈ greedy,
collapsing adv_std to 0.69, PPO ratios all ≈ 1.0.

### adv_std (Advantage Standard Deviation)

| adv_std | Interpretation |
|---------|---------------|
| > 1.5 | Strong learning signal — early training |
| 0.8 – 1.5 | Healthy mid-training |
| 0.5 – 0.8 | Weakening — monitor |
| < 0.5 | Near-collapse — policy not exploring enough |

### Entropy Health

| entropy | Interpretation |
|---------|---------------|
| > 1.0 | Too random — may not exploit learned policy |
| 0.5 – 1.0 | **Healthy range** — good exploration/exploitation |
| 0.3 – 0.5 | Early collapse — raise entropy_coef |
| < 0.3 | Collapsed — policy deterministic, no more learning |

**Phase 1 run (ENTROPY_COEF=0.05):** entropy = 0.546 at ep200 — never collapsed.
**Previous runs (ENTROPY_COEF=0.02, eta_min=1e-6):** entropy = 0.41 — collapsed at ep58.
**Current target (ENTROPY_COEF=0.01, eta_min=1e-5):** maintain 0.5–0.8 while allowing
policy to commit more, generating larger advantages.

### Loss Not Decreasing

| Observation | Likely Cause | Fix |
|------------|-------------|-----|
| clip_fraction near 0 | Entropy too high, adv signal zero | Reduce entropy_coef |
| Loss flat from start | LR too low or no gradients | Check lr, verify requires_grad |
| Loss → NaN | Numerical instability | Check log(0), div by zero |
| Value loss huge | Critic underfitting | Increase critic hidden dim |
| Entropy collapses to 0 | Policy collapsed | Increase entropy_coef |
| adv_std collapses | Sampled ≈ greedy | Reduce entropy_coef OR increase batch_size |

### Tour Length Not Improving

| Observation | Likely Cause | Fix |
|------------|-------------|-----|
| Stays ~random | Encoder not learning | Check gradient flow through encoder |
| 90% improvement in first 50 epochs, then flat | Advantage signal collapse | Reduce entropy_coef; increase batch_size |
| Best tour at final epoch | Model still learning at end of run | Increase n_epochs |
| Plateaus after improvement | LR too low | Check eta_min (must be 1e-5 not 1e-6) |

---

## 4. Chart Rendering Bugs (Fixed in v8 Phase 1b)

### Symptom
Entropy + entropy loss panel and Clip fraction + Lambda panel show overflowing tick labels —
dense illegible numbers spilling outside the panel boundaries.

### Root Cause
`matplotlib twinx()` auto-scales both axes independently. When the two axes have very
different value ranges (e.g. entropy 0.5–1.3 vs entropy_loss 0.025–0.067), matplotlib
generates hundreds of tick marks that overflow the figure boundary.

### Fix Applied
```python
import matplotlib.ticker

# Entropy panel — explicit ylim on both axes
a21.set_ylim(0.0, max(1.6, max(ent_hist) * 1.1) if ent_hist else 1.6)
ax_el.set_ylim(0.0, max(eloss_hist) * 1.15)
ax_el.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))

# Clip/Lambda panel — explicit ylim on both axes
a31.set_ylim(0.0, max(clip_hist) * 1.2 if max(clip_hist) > 0 else 0.1)
a31.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))
lam_pad = max((lam_max - lam_min) * 0.15, 0.1)
ax_lam.set_ylim(lam_min - lam_pad, lam_max + lam_pad)
ax_lam.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))
```

---

## 5. Path / Launch Error

### Symptom
```
FileNotFoundError: 'QAP-CVRP\\cvrp-ppo\\datasets\\val_n20.pkl'
```

### Root Cause
Script launched from the parent directory with a relative path argument.
Python resolves `__file__` as a relative path, so `SCRIPT_DIR` becomes relative,
breaking all dataset path construction.

### Fix — always cd into cvrp-ppo first
```
cd D:\Coding\RubyThesis\QAP-CVRP\cvrp-ppo
python train_n20.py
```

Or on Machine A:
```
cd "C:\Users\ASUS\Downloads\Thesis Code QAP_VRP\cvrp-ppo"
python train_n20.py
```

---

## 6. Validation Checklist

Run these checks after implementing each component:

### Encoder Checks (QAP mode only)
```python
norms = psi_prime.norm(dim=-1)
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "norm violation"
assert features.shape == (B, N+1, 5)
assert psi_prime.shape == (B, N+1, 2)
assert features[0, 0, 0].item() == 0.0, "depot demand ratio should be 0"
assert features[:, :, 4].abs().max() <= 1.0, "angle feature should be in [-1, 1]"
```

### Decoder Checks
```python
assert mask[:, 0].sum() == 0, "depot should never be masked"
assert (log_probs.exp()[mask] < 1e-6).all(), "infeasible nodes have nonzero prob"
```

### Tour Feasibility
```python
for b in range(B):
    tour = tours[b]
    total_demand = 0
    for node in tour:
        if node == 0: total_demand = 0
        else:
            total_demand += demands[b, node]
            assert total_demand <= capacity
    customers = [n for n in tour if n != 0]
    assert len(set(customers)) == N
```

### PPO Health Checks
```python
assert not torch.isnan(advantages).any()
assert advantages.std() > 0
ratio = (new_log_probs - old_log_probs).exp()
assert (ratio > 0).all()
assert ratio.max() < 100
# Healthy clip fraction: 2-15% of steps
clip_frac = ((ratio < 0.8) | (ratio > 1.2)).float().mean().item()
assert 0.001 < clip_frac < 0.30, f"Clip fraction {clip_frac:.3f} outside healthy range"
```

### OR-Tools Banner Checks
After `ensure_ortools_ref()` runs, verify it prints:
- Mean tour length
- Std + CV%
- ±2σ expected range
- 5% gap target (mean × 1.05)
- Current best model gap (if train_log.jsonl exists)

---

## 7. Debugging Workflow

When encountering an error:

1. **Identify the component:** encoder / decoder / environment / PPO / data / device / chart
2. **Check P1-P14** (Section 2 above) — P13/P14 for CUDA, path error for dataset not found
3. **Verify shapes** against Section 1 cheat sheet
4. **Check clip_fraction** in train_log.jsonl — near-zero means advantage collapse
5. **Minimal reproduction:** Reduce to B=2, N=5 and step through manually
6. **Check launch directory** — must run from inside cvrp-ppo/

### Print Debug Template
```python
def debug_step(state, action, scores, mask, psi_prime, query, device):
    print(f"Step {state.step}:")
    print(f"  clip_fraction should be 2-15%")
    print(f"  adv_std should be > 0.8")
    print(f"  entropy should be 0.5-1.0")
    print(f"  Current node : {state.current_node[0].item()}")
    print(f"  Remaining cap: {state.remaining_cap[0].item():.2f}")
    print(f"  Action       : {action[0].item()}")
    print(f"  Lambda       : {scoring.lambda_param.item():.4f}")
    if device.type == "cuda":
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM used    : {used:.2f} GB")
```
