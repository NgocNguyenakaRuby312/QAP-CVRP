# KNOWLEDGE: Debugging Guide & Common Pitfalls

**Purpose:** Quick-reference for diagnosing and fixing common issues in QAP-DRL implementation.
**When to reference:** Any error, unexpected behavior, NaN/Inf, shape mismatch, or convergence problem.
**Last updated:** May 2026

---

## 1. Shape Mismatch Cheat Sheet

Expected tensor shapes at every key point in the forward pass:

```
INPUT
  coords:           [B, N+1, 2]
  demands:          [B, N+1]        (depot demand = 0)
  capacity:         scalar float

FEATURE CONSTRUCTION
  features:         [B, N+1, 5]     order: [d/C, dist_depot, x, y, angle/pi]

ENCODER — QAP mode
  psi (after proj):     [B, N+1, 2]     unit norm!
  theta (from MLP):     [B, N+1]
  psi_prime (rotated):  [B, N+1, 2]     still unit norm!

ENCODER — baseline mode (ablation)
  embedding:            [B, N+1, 2]     NOT unit norm (ReLU output)

KNN
  knn_indices:      [B, N+1, k]     k=10 for CVRP-20, k=5 for CVRP-50/100

DECODER (per step t)
  psi_curr:         [B, 2]          current node amplitude (zero if at depot)
  context:          [B, 6]          [psi_curr(2), cap_ratio(1), step_ratio(1), x_curr(1), y_curr(1)]
  query:            [B, 2]          W_q * context  (W_q is 2x6, no bias)
  current_coords:   [B, 2]          vehicle Euclidean position (returned by context_query)
  context_scores:   [B, N+1]        query . psi_j
  interference:     [B, N+1]        sum_kNN psi_i . psi_j
  dist_to_nodes:    [B, N+1]        norm(coords_j - current_coords)
  scores:           [B, N+1]        context_scores + lambda*interference - mu*dist_to_nodes
  log_probs:        [B, N+1]        log_softmax(masked scores)
  action:           [B]             selected node index

CRITIC (psi_prime DETACHED)
  pooled:           [B, 2]          mean pool of psi_prime
  value:            [B]             MLP(pooled)

PPO
  advantages:       [B, T]
  returns:          [B, T]
  old_log_probs:    [B, T]

evaluate_actions() PPO update path
  ctx:              [mb, T, 6]
  query:            [mb, T, 2]
  cur_coords_3d:    [mb, T, 2]
  dist_to_nodes:    [mb, T, N+1]
  scores:           [mb, T, N+1]
```

---

## 2. Top 16 Known Pitfalls

### P1: Forgetting L2 Normalization After Projection
**Symptom:** Amplitude vectors have varying norms; rotation changes magnitude; scores explode.
**Fix:** Always apply `F.normalize(projected, p=2, dim=-1)` immediately after `self.proj(x)`.
**Verify:** `assert torch.allclose(psi.norm(dim=-1), torch.ones_like(psi[:,:,0]), atol=1e-5)`
**Note:** Only applies in QAP mode. Baseline encoder intentionally has no L2 norm.

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
**Fix:** Verify feature order: `[d/C, dist_depot, x, y, angle/pi]` (5D).

### P12: Angle Not Normalized by pi
**Symptom:** Feature index [4] has range [-pi, pi] instead of [-1, 1]; dominates other features.
**Fix:** Divide atan2 output by `math.pi`.

### P13: Tensor on Wrong Device (CUDA/CPU Mismatch)
**Symptom:** `RuntimeError: Expected all tensors to be on the same device`
**Fix:** `.to(device)` on all tensors at generation time. Never hardcode "cuda"/"cpu".

### P14: CUDA Out of Memory (OOM) on RTX 3050
**Symptom:** `torch.cuda.OutOfMemoryError`
**Fix sequence:**
```python
batch_size = 256
torch.cuda.empty_cache()
with torch.no_grad():
    results = evaluate(model, val_data)
```

### P15: context_query Returns Tuple — Not Unpacked (Change 2)
**Symptom:** `TypeError: cannot unpack non-sequence Tensor` or wrong shapes downstream.
**Root cause:** After Change 2, `ContextAndQuery.forward()` returns `(query, current_coords)`.
Any caller doing `query = self.context_query(...)` now has query as a 2-tuple, not a tensor.
**Fix:** Always unpack: `query, current_coords = self.context_query(state, psi_prime, step, n)`
**Affected callers:** `qap_decoder.py` forward(), `qap_policy.py` evaluate_actions().

### P16: mu_param Missing or Not Learnable (Change 1)
**Symptom:** Distance penalty has no effect; KeyError on 'mu_param'; mu stays at init value forever.
**Root cause:** `mu_param` defined as plain float or non-Parameter tensor in HybridScoring.
**Fix:** `self.mu_param = nn.Parameter(torch.tensor(0.5))` — must be `nn.Parameter`.
**Verify:**
```python
assert hasattr(model.decoder.hybrid, 'mu_param')
assert any(p is model.decoder.hybrid.mu_param for p in model.parameters())
```

---

## 3. Training Convergence Diagnostics

### Clip Fraction is the Key Indicator

| clip_fraction | Interpretation | Action |
|--------------|----------------|--------|
| < 0.1% | Policy not updating — advantage signal too weak | Reduce entropy_coef; increase batch_size |
| 0.1% - 2% | Weak updates — borderline | Monitor adv_std |
| 2% - 15% | Healthy range | No action needed |
| 15% - 30% | Policy changing fast | Reduce LR or increase ppo_epochs |
| > 30% | Over-clipping — policy unstable | Reduce LR drastically |

Phase 1 run had clip_fraction = 0.014% (mean) — critically low.
Root cause: ENTROPY_COEF=0.05 kept entropy at H=0.55, making sampled approx greedy,
collapsing adv_std to 0.69, PPO ratios all near 1.0.

### adv_std (Advantage Standard Deviation)

| adv_std | Interpretation |
|---------|---------------|
| > 1.5 | Strong learning signal — early training |
| 0.8 - 1.5 | Healthy mid-training |
| 0.5 - 0.8 | Weakening — monitor |
| < 0.5 | Near-collapse — policy not exploring enough |

### Entropy Health

| entropy | Interpretation |
|---------|---------------|
| > 1.0 | Too random |
| 0.5 - 1.0 | Healthy range |
| 0.3 - 0.5 | Early collapse — raise entropy_coef |
| < 0.3 | Collapsed |

### mu (Distance Penalty) Behaviour
After training starts, mu should drift from 0.5 toward values that balance proximity
vs amplitude coherence for the specific problem size. Healthy range: 0.2-2.0.
- mu to 0: policy relying entirely on amplitude geometry (not enough proximity bias)
- mu > 2: policy becoming greedy-nearest-neighbour (ignoring interference)
Log mu_val in train_log.jsonl to track.

### Loss Not Decreasing

| Observation | Likely Cause | Fix |
|------------|-------------|-----|
| clip_fraction near 0 | Entropy too high, adv signal zero | Reduce entropy_coef |
| Loss flat from start | LR too low or no gradients | Check lr, verify requires_grad |
| Loss to NaN | Numerical instability | Check log(0), div by zero |
| Value loss huge | Critic underfitting | Increase critic hidden dim |
| Entropy collapses to 0 | Policy collapsed | Increase entropy_coef |
| adv_std collapses | Sampled approx greedy | Reduce entropy_coef OR increase batch_size |

---

## 4. Chart Rendering Bugs (Fixed)

### Tick label overflow (Fixed v8 Phase 1b)
**Symptom:** Entropy + entropy loss panel and Clip fraction + Lambda panel show overflowing tick labels.
**Fix:**
```python
import matplotlib.ticker
ax_el.set_ylim(0.0, max(eloss_hist) * 1.15)
ax_el.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))
```

### Dense grey gridlines on twinx panels (Fixed v9)
**Symptom:** Dual-axis panels filled with dense grey horizontal lines.
**Root cause:** seaborn whitegrid style applies to both primary and twinx axes.
**Fix:** Add 3 lines after every `twinx()` call:
```python
ax_twin.set_zorder(primary.get_zorder() - 1)
ax_twin.patch.set_visible(False)
ax_twin.grid(False)
```

---

## 5. Path / Launch Error

### Symptom
```
FileNotFoundError: 'QAP-CVRP\\cvrp-ppo\\datasets\\val_n20.pkl'
```

### Fix
```
cd "C:\Users\ASUS\Downloads\Thesis Code QAP_VRP\cvrp-ppo"
python train_n20.py
```

---

## 6. Evaluation Bugs (Fixed)

### P18: val_tour > greedy_tour = augmentation bug
**Symptom:** val_tour(aug×8) worse than greedy_tour.
**Root cause:** stochastic augmentation with a static encoder is still valid, but if stochastic sampling was accidentally used instead of coordinate augmentation + greedy, val_tour will be inflated.
**Fix:** evaluate.py v3 (coordinate augmentation + greedy). After fix, val_tour should be ≤ greedy_tour.

### P19: Dense grey gridlines on twinx() chart panels
See Section 4 above for fix.

### Symptom (Path error)
See Section 5 above for fix.

---

## 7. Validation Checklist

### Encoder Checks (QAP mode only)
```python
norms = psi_prime.norm(dim=-1)
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "norm violation"
assert features.shape == (B, N+1, 5)
assert psi_prime.shape == (B, N+1, 2)
assert features[0, 0, 0].item() == 0.0, "depot demand ratio should be 0"
assert features[:, :, 4].abs().max() <= 1.0, "angle feature should be in [-1, 1]"
assert features.shape[-1] == 5, "must be 5D"
```

### Decoder Checks (with Changes 1+2)
```python
assert mask[:, 0].sum() == 0, "depot should never be masked"
assert (log_probs.exp()[mask] < 1e-6).all(), "infeasible nodes have nonzero prob"
# Change 2
assert ctx.shape[-1] == 6, "context must be 6D after Change 2"
assert current_coords.shape == (B, 2), "current_coords must be returned by context_query"
# Change 1
assert dist_to_nodes.shape == (B, N+1), "dist_to_nodes must exist"
assert hasattr(model.decoder.hybrid, 'mu_param'), "mu_param missing from HybridScoring"
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
clip_frac = ((ratio < 0.8) | (ratio > 1.2)).float().mean().item()
assert 0.001 < clip_frac < 0.30, f"Clip fraction {clip_frac:.3f} outside healthy range"
```

---

## 8. Debugging Workflow

When encountering an error:

1. **Identify the component:** encoder / decoder / environment / PPO / data / device / chart
2. **Check P1-P16** — P13/P14 for CUDA, P15 for tuple unpack, P16 for mu_param
3. **Verify shapes** against Section 1 cheat sheet (context must be [B,6] not [B,4])
4. **Check clip_fraction** in train_log.jsonl — near-zero means advantage collapse
5. **Minimal reproduction:** Reduce to B=2, N=5 and step through manually
6. **Check launch directory** — must run from inside cvrp-ppo/

### Print Debug Template
```python
def debug_step(state, action, scores, mask, psi_prime, query, mu, device):
    print(f"Step {state.step}:")
    print(f"  Current node : {state.current_node[0].item()}")
    print(f"  Remaining cap: {state.remaining_cap[0].item():.2f}")
    print(f"  Action       : {action[0].item()}")
    print(f"  Lambda       : {model.decoder.hybrid.lambda_param.item():.4f}")
    print(f"  Mu           : {model.decoder.hybrid.mu_param.item():.4f}")
    if device.type == "cuda":
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM used    : {used:.2f} GB")
```
