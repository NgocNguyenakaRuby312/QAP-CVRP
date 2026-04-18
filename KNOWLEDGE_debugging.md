# KNOWLEDGE: Debugging Guide & Common Pitfalls

**Purpose:** Quick-reference for diagnosing and fixing common issues in QAP-DRL implementation.
**When to reference:** Any error, unexpected behavior, NaN/Inf, shape mismatch, or convergence problem.
**Last updated:** March 2026

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

ENCODER (§3.X.4-5)
  psi (after proj):     [B, N+1, 2]     ← unit norm!
  theta (from MLP):     [B, N+1]
  psi_prime (rotated):  [B, N+1, 2]     ← still unit norm!

KNN
  knn_indices:      [B, N+1, k]     where k=5 default, no self-loops

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
  pooled:           [B, 2]          ← mean pool of psi_prime
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
**Fix:** Episode length = exactly N customer selections + depot returns. Track carefully. Pad shorter episodes in the buffer.

### P7: Capacity Not Reset on Depot Return
**Symptom:** After returning to depot, vehicle still has reduced capacity; customers become infeasible.
**Fix:** In `env.step()`, when action == 0 (depot): `remaining_cap = capacity`.

### P8: Demand of Depot Not Zero
**Symptom:** Visiting depot reduces capacity; cascading feasibility issues.
**Fix:** Ensure `demands[:, 0] = 0` during data generation and never modify it.

### P9: Gradient Through Sampling
**Symptom:** Gradients are None or zero for the policy.
**Fix:** Use `torch.distributions.Categorical` for sampling, collect `log_prob(action)` for PPO. Don't use `argmax` during training (no gradient).

### P10: kNN Indices Stale After Clustering
**Symptom:** kNN references nodes from other clusters; index out of bounds.
**Fix:** Recompute kNN per sub-problem after clustering. kNN is always relative to the current node set.

### P11: Feature Order Mismatch
**Symptom:** Model trains but performs poorly; amplitude projection learns wrong feature mapping.
**Fix:** Verify feature order matches thesis §3.X.3: `[d/C, dist, x, y, angle/π]`.
**Verify:** Print `features[0, 1]` for a known node and manually check each component.

### P12: Angle Not Normalized by π
**Symptom:** Feature index [4] has range [-π, π] instead of [-1, 1]; dominates other features.
**Fix:** Divide atan2 output by `math.pi` — thesis §3.X.3 specifies `αᵢ/π`.
**Verify:** `assert features[:, :, 4].abs().max() <= 1.0`

### P13: Tensor on Wrong Device (CUDA/CPU Mismatch)
**Symptom:** `RuntimeError: Expected all tensors to be on the same device` or
`RuntimeError: Tensor for argument #2 'mat1' is on CPU, but expected it to be on GPU`.
**Fix:**
```python
# Check at module boundaries
assert coords.device == demands.device, "coords/demands device mismatch"
assert next(model.parameters()).device == coords.device, "model/data device mismatch"

# Fix: ensure all tensors go to device at generation time
coords  = coords.to(device)
demands = demands.to(device)
knn_indices = knn_indices.to(device)
```
**Root cause:** Tensor created on CPU (default) but model is on CUDA. Always pass `device` to data_generator and knn functions. Never call `.to(device)` inside a module — do it externally in run.py.

### P14: CUDA Out of Memory (OOM) on RTX 3050
**Symptom:** `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB`.
**Hardware limit:** RTX 3050 has 4GB VRAM. ~3.6GB usable after system overhead.
**Fix sequence:**
```python
# Step 1: reduce batch size
batch_size = 128   # from 256

# Step 2: clear cache at epoch start
torch.cuda.empty_cache()

# Step 3: ensure no_grad during evaluation
with torch.no_grad():
    results = evaluate(model, val_data)

# Step 4: monitor usage
used  = torch.cuda.memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {used:.2f} / {total:.2f} GB")
```
**Do NOT:** Increase batch_size above 256 for CVRP-100 without testing first.

---

## 3. Convergence Troubleshooting

### Loss Not Decreasing

| Observation | Likely Cause | Fix |
|------------|-------------|-----|
| Loss stays flat from start | LR too low or no gradients | Check lr (try 3e-4); verify `requires_grad=True` |
| Loss oscillates wildly | LR too high or batch too small | Reduce lr (try 3e-5); increase batch_size |
| Loss decreases then plateaus | PPO clip too tight or entropy low | Increase `eps_clip` to 0.3; increase `c2` to 0.05 |
| Loss goes to NaN | Numerical instability | Check for log(0), division by zero; use `log_softmax` |
| Value loss huge, policy okay | Critic underfitting | Increase critic hidden dim; higher lr for critic |
| Entropy collapses to 0 | Policy collapsed | Increase `c2`; check temperature |
| Only λ or W_q has gradient | Gradient blocked | Verify psi_prime is not detached; check shared encoder |

### Tour Length Not Improving

| Observation | Likely Cause | Fix |
|------------|-------------|-----|
| Stays ~random | Encoder not learning | Check psi vectors aren't identical; verify gradient flow |
| Good CVRP-20, bad CVRP-50 | Capacity handling bug | Verify capacity reset; check masking for larger N |
| Plateaus after improvement | kNN not contributing | Check λ gradient; try different k; verify knn_indices |
| Tours have capacity violations | Masking bug | Print mask at each step |
| All tours same order | Policy collapsed | Increase c2; verify sampling (not greedy) during training |
| Improves then worsens | Overfitting or LR too high | Add LR decay; validate regularly |

---

## 4. Validation Checklist

Run these checks after implementing each component:

### Encoder Checks
```python
# Unit norm verification
features = feature_constructor(coords, demands, capacity)
psi_prime = encoder(features)
norms = psi_prime.norm(dim=-1)
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
    f"Norm violation: min={norms.min():.6f}, max={norms.max():.6f}"

# Shape verification
assert features.shape == (B, N+1, 5)
assert psi_prime.shape == (B, N+1, 2)

# Feature order verification
assert features[0, 0, 0].item() == 0.0, "Depot demand ratio should be 0"
assert features[:, :, 4].abs().max() <= 1.0, "Angle feature should be in [-1, 1]"

# Gradient flow through full encoder
loss = psi_prime.sum()
loss.backward()
for name, p in encoder.named_parameters():
    assert p.grad is not None, f"No gradient for {name}"
    assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

# Device check
assert psi_prime.device.type == device.type, "Encoder output on wrong device"
```

### Decoder Checks
```python
# Masking verification
mask = get_feasibility_mask(visited, demands, remaining_cap)
assert mask[:, 0].sum() == 0, "Depot should never be masked"
assert (log_probs.exp()[mask] < 1e-6).all(), "Infeasible nodes have nonzero prob"

# Tour feasibility
for b in range(B):
    tour = tours[b]
    total_demand = 0
    for node in tour:
        if node == 0:
            total_demand = 0
        else:
            total_demand += demands[b, node]
            assert total_demand <= capacity, f"Capacity violation at node {node}"
    customer_visits = [n for n in tour if n != 0]
    assert len(set(customer_visits)) == N, "Not all customers visited"
    assert len(customer_visits) == N, "Duplicate visits"
```

### PPO Checks
```python
# Advantage computation
assert not torch.isnan(advantages).any(), "NaN in advantages"
assert advantages.std() > 0, "Zero-variance advantages"

# Ratio check
ratio = (new_log_probs - old_log_probs).exp()
assert (ratio > 0).all(), "Negative ratio — log_prob sign error"
assert ratio.max() < 100, f"Extreme ratio {ratio.max():.2f}"
```

### Device Checks
```python
# All tensors on same device
assert coords.device.type == device.type,        "coords on wrong device"
assert demands.device.type == device.type,       "demands on wrong device"
assert knn_indices.device.type == device.type,   "knn_indices on wrong device"
assert next(model.parameters()).device.type == device.type, "model on wrong device"

# VRAM usage after each phase (if CUDA)
if device.type == "cuda":
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM after phase: {used:.2f} / {total:.2f} GB")
```

### Full Pipeline Smoke Test
```python
# Tiny end-to-end test: B=2, N=5, train 20 steps
model = QAPPolicy(opts).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
instance = generate_instance(B=2, N=5, capacity=15, device=device)
rewards_start = evaluate(model, instance, greedy=True).mean().item()
for _ in range(20):
    train_step(model, optimizer, instance)
rewards_end = evaluate(model, instance, greedy=True).mean().item()
assert rewards_end > rewards_start, "Model did not improve on trivial instance"
print(f"Smoke test passed on device: {device}")
```

---

## 5. PyTorch Version Compatibility

### PyTorch 2.0+ with CUDA (RTX 3050)
Install with cu121 build:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Verify:
```python
import torch
assert torch.cuda.is_available(), "CUDA not available — reinstall cu121 build"
assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3050"
```

### PyTorch 2.1+ Specific Issues
- `ReduceLROnPlateau`: `verbose` kwarg deprecated. Use `verbose=False` or remove.
- `torch.cuda.amp` moved to `torch.amp`. Update imports if using mixed precision.
- `torch.compile()` may break custom autograd. Disable if hitting unexplained errors.

### Common Import Patterns
```python
# CORRECT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# WRONG — never import these
# import pennylane  ← NO quantum libraries
# import qiskit     ← NO quantum libraries
```

---

## 6. Performance Profiling

### Bottleneck Locations (by frequency)
1. **kNN computation** — O(N² · B). Precompute once per instance, cache for all decode steps.
2. **Autoregressive decoding** — N sequential steps, can't parallelize across steps.
3. **Interference term gather** — `torch.gather` with expanded indices. Pre-expand once, reuse.
4. **PPO update** — K=3 epochs over full buffer. Minibatch if memory-constrained.

### Memory Optimization (RTX 3050 specific)
```python
# Estimated VRAM usage
# CVRP-20,  B=256: ~0.5 GB  ← safe
# CVRP-50,  B=256: ~1.0 GB  ← safe
# CVRP-100, B=256: ~2.0 GB  ← safe
# CVRP-100, B=512: ~4.0 GB  ← borderline, test first

# If OOM: reduce batch_size first
batch_size = 128

# Clear cache at epoch boundaries
torch.cuda.empty_cache()

# Use int32 for kNN indices to save VRAM
knn_indices = knn_indices.to(torch.int32)
```

### Speed Tips
- Move kNN to GPU: `torch.cdist` is GPU-accelerated
- Use `torch.no_grad()` for all evaluation and greedy rollout
- Pre-allocate tour tensors: `tours = torch.zeros(B, max_steps, dtype=torch.long, device=device)`
- Cache `psi_prime` — it doesn't change during decoding

---

## 7. Debugging Workflow

When encountering an error:

1. **Identify the component:** encoder / decoder / environment / PPO / data / device
2. **Check P1-P14** (Section 2 above) — especially P13/P14 for CUDA errors
3. **Verify shapes** against Section 1 cheat sheet
4. **Run component test** from `tests/` folder
5. **Add assertions** at suspected failure point
6. **Minimal reproduction:** Reduce to B=2, N=5, step through manually
7. **Check device:** `print(tensor.device)` at every module boundary

### Print Debug Template
```python
def debug_step(state, action, scores, mask, psi_prime, query, device):
    print(f"Step {state.step}:")
    print(f"  Device       : {device}")
    print(f"  Current node : {state.current_node[0].item()}")
    print(f"  Remaining cap: {state.remaining_cap[0].item():.2f}")
    print(f"  Visited      : {state.visited[0].nonzero().squeeze().tolist()}")
    print(f"  Action       : {action[0].item()}")
    print(f"  Mask (infeas): {mask[0].nonzero().squeeze().tolist()}")
    print(f"  Top-3 scores : {scores[0].topk(3)}")
    print(f"  Query norm   : {query[0].norm().item():.4f}")
    print(f"  Psi_prime norm: min={psi_prime[0].norm(dim=-1).min():.4f}, "
          f"max={psi_prime[0].norm(dim=-1).max():.4f}")
    print(f"  Lambda       : {scoring.lambda_param.item():.4f}")
    if device.type == "cuda":
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM used    : {used:.2f} GB")
```
