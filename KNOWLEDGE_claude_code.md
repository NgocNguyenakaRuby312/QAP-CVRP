# KNOWLEDGE_claude_code.md — QAP-DRL Quick Reference for Claude Code

## Project Purpose

Implement the QAP-DRL (Quantum-Amplitude Perturbation Deep Reinforcement Learning) framework
for CVRP as a constructive solver. Based on the thesis: "Quantum-Amplitude Perturbation Deep
Reinforcement Learning for Clustered Vehicle Routing Problems" (IU HCMC, 2025).

**Golden rule:** Implement exactly as specified. No improvements, no extras.

---

## Critical Context

**Paradigm:** CONSTRUCTIVE (build solution from scratch, NOT improvement/refinement)
**Base reference:** VRP-DACT repo structure (in `ref_code/`), but architecture is completely different
**Target folder:** `cvrp-ppo/` — all implementation goes here
**Language:** Python 3.10+, PyTorch ≥ 2.0 (cu121 CUDA build)
**Hardware:** NVIDIA GeForce RTX 3050, 4GB VRAM, CUDA 13.2

---

## Architecture Summary

```
5D Features [d/C, dist, x, y, angle/π]
    → Linear(5→2) + L2Norm → ψ on unit circle
    → Rotation(MLP(5→16→1, tanh)→θ→R(θ)·ψ) → ψ' on unit circle
    → Autoregressive decoder: context query + hybrid scoring (attention + kNN interference)
    → PPO training (ε=0.2, K=3, γ=0.99, GAE λ=0.95)
```

### Encoder (4 components)
1. **FeatureConstructor:** `[d/C, dist, x, y, angle/π]` → [B, N+1, 5]
2. **AmplitudeProjection:** `Linear(5→2) + F.normalize` → [B, N+1, 2] (unit norm!)
3. **RotationMLP:** `MLP(5→16→1, tanh)` → θ_i per node → [B, N+1]
4. **Rotation:** `R(θ_i) · ψ_i` → ψ'_i → [B, N+1, 2] (norm preserved)

### Decoder (autoregressive, N steps)
5. **ContextQuery:** `[ψ'_curr, cap_ratio, step_ratio]` → `W_q(4→2, no bias)` → query [B, 2]
6. **HybridScoring:** `q·ψ'_j + λ·Σ_{kNN}(ψ'_i·ψ'_j)` → scores [B, N+1]
7. **Masking:** infeasible → -1e9, depot ALWAYS feasible → log_softmax → sample/greedy

### Critic
- Shared encoder, mean-pool ψ' → MLP(2→64→1) → V(s)

### Parameter Budget
- Amplitude W+b: 12
- Rotation MLP (5→16→1): ~113
- Query W_q: 8
- λ: 1
- Critic: ~257
- **Actor total: ~134, Full total: ~391**

---

## Implementation Rules

1. PyTorch only — NO PennyLane, Qiskit, quantum libraries
2. Batch-first: `[batch_size, N+1, ...]`
3. ALWAYS L2-normalize after amplitude projection
4. Mask BEFORE softmax (set -1e9), NEVER after
5. Depot (index 0) NEVER masked
6. kNN precomputed per instance from spatial coords, no self-loops
7. Shared encoder between actor and critic
8. No curriculum learning, no data augmentation at inference
9. demands ∈ [1,9] integers; depot demand = 0
10. Episode ends when ALL N customers visited
11. Feature order: [d/C, dist, x, y, angle/π] (thesis §3.X.3)
12. Angle normalized by π (range [-1, 1])
13. Zero vector for ψ'_curr when at depot (thesis §3.X.6)
14. ALL tensors and models must be on `device` — never hardcode "cuda" or "cpu"

---

## Hyperparameters (options.py)

```
graph_size: 20/50/100    capacity: 30/40/50
embedding_dim: 2         hidden_dim: 16        knn_k: 5
lambda_init: 0.1         K_epochs: 3           eps_clip: 0.2
gamma: 0.99              gae_lambda: 0.95      c1: 0.5     c2: 0.01
lr: 1e-4                 batch_size: 256       epoch_size: 128000
n_epochs: 100-200        max_grad_norm: 1.0    val_size: 10000
```

---

## File Structure

```
cvrp-ppo/
├── run.py
├── options.py
├── encoder/           ← feature_constructor, amplitude_projection, rotation_mlp, rotation, qap_encoder
├── decoder/           ← context_query, hybrid_scoring, qap_decoder
├── environment/       ← cvrp_env, state
├── models/            ← qap_policy
├── training/          ← ppo_agent, rollout_buffer, evaluate
├── utils/             ← knn, clustering, data_generator, seed, logger, checkpoint, metrics
├── tests/             ← test_encoder, test_decoder, test_env, test_smoke
├── datasets/
└── outputs/
```

---

## Validation Checks

After any code change, verify:
- [ ] ψ' vectors: L2 norm = 1.0 (atol=1e-5)
- [ ] All tours feasible (no capacity violations)
- [ ] All N customers visited exactly once
- [ ] Depot at start/end of every sub-route
- [ ] Score shape: [B, N+1] before masking
- [ ] kNN: no self-loops, computed from spatial coords
- [ ] Feature order: [d/C, dist, x, y, angle/π]
- [ ] Angle feature in [-1, 1] range
- [ ] Greedy CVRP-20 produces valid tours
- [ ] PPO loss decreases in early training
- [ ] Parameter count matches budget
- [ ] All tensors on correct device (P13)
- [ ] No OOM errors with batch_size=256 (P14)

---

## CC Workflow

1. Read this file first
2. Check implementation order in KNOWLEDGE_codebase.md
3. Implement component by component, test each independently
4. Full files only — no partial snippets
5. Include shape comments on every tensor operation
6. Run smoke test after each phase
7. Always pass `device` as argument — detect only in run.py

---

## Data Generation

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coords  = torch.FloatTensor(B, N+1, 2).uniform_(0, 1).to(device)
demands = torch.zeros(B, N+1, dtype=torch.long, device=device)
demands[:, 1:] = torch.randint(1, 10, (B, N), device=device)
capacity = {20: 30, 50: 40, 100: 50}[N]
```

---

## GPU / CUDA Setup

**Hardware:** NVIDIA GeForce RTX 3050, 4GB VRAM, CUDA 13.2

### Verify CUDA
```python
import torch
print(torch.cuda.is_available())           # must be True
print(torch.cuda.get_device_name(0))       # NVIDIA GeForce RTX 3050
```

If False — reinstall:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Device Pattern
```python
# run.py ONLY — detect once, pass everywhere as argument
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
if device.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
```

### VRAM Budget

| Problem | batch_size | VRAM | Status |
|---------|-----------|------|--------|
| CVRP-20 | 256 | ~0.5 GB | ✓ Safe |
| CVRP-50 | 256 | ~1.0 GB | ✓ Safe |
| CVRP-100 | 256 | ~2.0 GB | ✓ Safe |
| CVRP-100 | 512 | ~4.0 GB | ⚠ Test first |

### OOM Recovery
```python
torch.cuda.empty_cache()   # start of every epoch
batch_size = 128            # if OOM at 256
```

### Training Time (RTX 3050)

| Problem | Per epoch | 100 epochs |
|---------|----------|------------|
| CVRP-20 | ~3–5 min | ~5–8 hrs |
| CVRP-50 | ~8–12 min | ~13–20 hrs |
| CVRP-100 | ~20–30 min | ~33–50 hrs |

Save checkpoints every epoch. Train CVRP-100 overnight.
