# KNOWLEDGE_claude_code.md — QAP-DRL Quick Reference for Claude Code

## Project Purpose

QAP-DRL (Quantum-Amplitude Perturbation Deep Reinforcement Learning) framework
for CVRP as a constructive solver. Thesis: IU HCMC, 2025.
Reference paper: Giang et al. (2025) — Q-GAT.

**Golden rule:** Implement exactly as specified. No improvements, no extras.

---

## Critical Context

**Paradigm:** CONSTRUCTIVE (build from scratch, NOT improvement)
**Target folder:** `cvrp-ppo/`
**Language:** Python 3.10+, PyTorch ≥ 2.0 (cu121 CUDA build)
**Hardware:** NVIDIA GeForce RTX 3050, 4GB VRAM
**Two machines:**
- Machine A: `C:\Users\ASUS\Downloads\Thesis Code QAP_VRP\cvrp-ppo\`
- Machine B: `D:\Coding\RubyThesis\QAP-CVRP\cvrp-ppo\`

---

## Architecture Summary (Phase 2: 4D)

```
5D Features [d/C, dist_depot, x, y, angle/π]   (STATIC — computed once)
    → Linear(5→4) + L2Norm → ψ on S³ hypersphere
    → Rotation(MLP(5→32→6, tanh) → 6 Givens angles → SO(4)·ψ) → ψ'
    → psi_prime FIXED for all decode steps
    → Context: [ψ'_curr(4), cap/C, t/N, x_curr, y_curr] → Wq(8→4) → query
    → Scoring: q·ψ'_j + λ·E_kNN(j) − μ·dist(vₜ,vⱼ) → softmax → action
    → PPO training (ε=0.2, K=3, γ=0.99, GAE λ=0.95)
    → CosineAnnealingWarmRestarts(T_0=50, T_mult=2)
```

### Encoder (2 variants)
**QAP mode (default):**
1. FeatureConstructor: [d/C, dist_depot, x, y, angle/π] → [B, N+1, 5]
2. AmplitudeProjection: Linear(5→4) + F.normalize → [B, N+1, 4] (unit norm on S³)
3. RotationMLP: MLP(5→32→6, tanh) → 6 Givens angles → [B, N+1, 6]
4. SO(4) Rotation: G₆·G₅·G₄·G₃·G₂·G₁·ψ → ψ' → [B, N+1, 4]
5. Called ONCE — psi_prime fixed for all decode steps

**Baseline mode (ablation):**
1. Same 5D features
2. Linear(5→4) + ReLU (STATIC — no rotation, no norm)

### Decoder (autoregressive, N steps)
5. ContextQuery: [ψ'_curr(4), cap/C, t/N, x_curr, y_curr] → Wq(8→4) → (query, current_coords)
6. HybridScoring: q·ψ'_j + λ·E_kNN(j) − μ·dist(vₜ,vⱼ) → scores [B, N+1]
7. Masking: infeasible → -1e9, depot ALWAYS feasible → log_softmax → sample

### Parameter Budget

| Model | Actor | Full |
|-------|-------|------|
| QAP-DRL (Phase 2: 4D) | ~288 | **~609** |
| Pure DRL baseline | ~56 | ~379 |

---

## Current Hyperparameters (Phase 2+)

```python
BATCH_SIZE    = 512
EPOCH_SIZE    = 128_000
ENTROPY_COEF  = 0.03
KNN_K         = 10 (N=20), 15 (N=50), 20 (N=100), 30 (N=200)
AUG_SAMPLES   = 8
AMP_DIM       = 4        # S³ hypersphere
HIDDEN_DIM    = 32       # rotation MLP
MU_INIT       = 0.5      # clamp [0, 20]
LAMBDA_INIT   = 0.1      # clamp [-2, 3]
Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-5)
No weight_decay on μ
```

---

## File Status

| File | Description | Status |
|------|-------------|--------|
| `encoder/feature_constructor.py` | Static 5D | ✓ |
| `encoder/amplitude_projection.py` | output_dim=4, W 4×5, S³ | ✓ |
| `encoder/rotation_mlp.py` | n_angles=6, 5→32→6 | ✓ |
| `encoder/rotation.py` | 6 Givens rotations for SO(4) | ✓ |
| `encoder/qap_encoder.py` | amp_dim=4, static | ✓ |
| `decoder/context_query.py` | ctx ℝ⁸, Wq 4×8, embed_dim=4 | ✓ |
| `decoder/hybrid_scoring.py` | dimension-agnostic, μ [0,20], λ [-2,3] | ✓ |
| `decoder/qap_decoder.py` | context_dim=8, embed_dim=4 | ✓ |
| `models/qap_policy.py` | amp_dim=4, critic 4→64→1 | ✓ |
| `training/ppo_agent.py` | WarmRestarts, no WD on μ | ✓ |
| `training/evaluate.py` | v3: coord-aug+greedy | ✓ |
| `train_n20.py` | Base script, HIDDEN_DIM=32 module-level | ✓ |
| `train_n50/100/200.py` | Import train_n20 + override settings | ✓ |
| `test_n20/50/100/200.py` | Standalone test scripts | ✓ |

---

## Implementation Rules

1. PyTorch only — NO PennyLane, Qiskit
2. Batch-first: `[batch_size, N+1, ...]`
3. QAP mode: ALWAYS L2-normalize after amplitude projection (S³)
4. Mask BEFORE softmax (-1e9), NEVER after
5. Depot (index 0) NEVER masked
6. kNN precomputed once per instance from spatial coords
7. Shared encoder between actor and critic
8. psi_prime DETACHED before critic head
9. Feature order: [d/C, dist_depot, x, y, angle/π] (5D)
10. ψ'_curr = zero vector(4D) at depot; x_curr,y_curr = actual depot coords
11. `context_query.forward()` returns TUPLE — always unpack
12. `decoder.rollout()` uses fixed psi_prime — no per-step re-encoding
13. `evaluate_actions()` broadcasts static psi_prime
14. ALL `QAPPolicy()` calls must pass `amp_dim=AMP_DIM, hidden_dim=HIDDEN_DIM`
15. Run train scripts from inside cvrp-ppo/

---

## Validation Checks

- [ ] `features.shape[-1] == 5` — 5D features
- [ ] `psi_prime.shape[-1] == 4` — 4D amplitudes on S³
- [ ] ψ' vectors: L2 norm = 1.0 (atol=1e-5)
- [ ] `ctx.shape[-1] == 8` — context is D+4=8
- [ ] `query.shape[-1] == 4` — query is 4D
- [ ] `current_coords.shape == (B, 2)` — returned by context_query
- [ ] `dist_to_nodes.shape == (B, N+1)`
- [ ] `mask[:, 0].sum() == 0` — depot never masked
- [ ] Policy creation includes `amp_dim` and `hidden_dim`

---

## Pitfalls Reference

| # | Pitfall | Fix |
|---|---------|-----|
| P1 | Missing L2 norm | `F.normalize(p=2, dim=-1)` on S³ |
| P2 | Mask after softmax | -1e9 BEFORE log_softmax |
| P3 | Depot masked | `mask[:, 0] = False` |
| P4 | kNN self-loops | `diagonal.fill_(inf)` before topk |
| P5 | NaN in rotation | `torch.clamp(theta, -10, 10)` |
| P13 | Wrong device | `.to(device)` everywhere |
| P14 | OOM RTX 3050 | batch_size=256; empty_cache() |
| P15 | context_query tuple | `query, current_coords = context_query(...)` |
| P16 | mu_param not Parameter | `nn.Parameter(torch.tensor(0.5))` |
| P17 | feature_dim=6 | Must be 5. Encoder is static 5D |
| P18 | Per-step re-encoding | NEVER. psi_prime fixed |
| P19 | evaluate_actions rebuilds psi_prime | BROADCAST static |
| P20 | Missing amp_dim/hidden_dim in reload | `QAPPolicy(..., amp_dim=4, hidden_dim=32)` |
| P21 | Optimizer state mismatch | Delete old epoch files |
| P24 | twinx gridlines | `.set_zorder(-1)`, `.patch.set_visible(False)`, `.grid(False)` |
