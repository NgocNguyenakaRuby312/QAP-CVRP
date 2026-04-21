# KNOWLEDGE_claude_code.md — QAP-DRL Quick Reference for Claude Code

## Project Purpose

Implement the QAP-DRL (Quantum-Amplitude Perturbation Deep Reinforcement Learning) framework
for CVRP as a constructive solver. Based on the thesis: "Quantum-Amplitude Perturbation Deep
Reinforcement Learning for Clustered Vehicle Routing Problems" (IU HCMC, 2025).

**Key reference paper:**
> Giang et al. (2025) — Q-GAT: real PQCs inside GAT encoder, ~50% param reduction vs standard GAT.
> QAP-DRL: same motivation, purely classical, no quantum simulator.

**Golden rule:** Implement exactly as specified. No improvements, no extras.

---

## Critical Context

**Paradigm:** CONSTRUCTIVE (build from scratch, NOT improvement/refinement)
**Target folder:** `cvrp-ppo/` — all implementation goes here
**Language:** Python 3.10+, PyTorch ≥ 2.0 (cu121 CUDA build)
**Hardware:** NVIDIA GeForce RTX 3050, 4GB VRAM, CUDA 13.2
**Two machines:**
- Machine A: `C:\Users\ASUS\Downloads\Thesis Code QAP_VRP\cvrp-ppo\`
- Machine B: `D:\Coding\RubyThesis\QAP-CVRP\cvrp-ppo\`
- **Always `cd` into cvrp-ppo before running**

---

## Methodology Changes Applied (May 2026) — Changes 1+2 ACTIVE, Change 3 REVERTED

Changes 1+2 are permanent canonical. Change 3 was reverted.

### Change 1 — Distance Proximity Penalty (hybrid_scoring.py)
```
Score(j) = q·ψ'ⱼ + λ·E_kNN(j) − μ·dist(vₜ, vⱼ)
```
- `self.mu_param = nn.Parameter(torch.tensor(0.5))` in HybridScoring
- `forward()` signature: `(query, psi_prime, knn_indices, mask, current_coords, all_coords)`
- +1 parameter

### Change 2 — Spatial Context Grounding (context_query.py)
```
ctx = [ψ'_curr(2), cap/C(1), t/N(1), x_curr(1), y_curr(1)]  ∈ ℝ⁶
query = Wq · ctx,   Wq ∈ ℝ^{2×6}
```
- `self.Wq = nn.Linear(6, 2, bias=False)`
- `forward()` returns `(query [B,2], current_coords [B,2])` — **always unpack both**
- +4 parameters

### Change 3 — Dynamic Proximity Feature — REVERTED
```
ATTEMPTED: xᵢ(t) = [d/C, dist_depot, x, y, angle/π, dist(i, vₜ)]  ∈ ℝ⁶
REVERTED: per-step re-encoding destabilized training catastrophically.
  λ → −1.6, μ → 3.4, val_tour → 13.7 (123% gap) in 22 epochs.
```
- Encoder is STATIC: 5D features, computed once, psi_prime fixed for all steps
- `FeatureBuilder.forward(state)` — single argument, returns `[B, N+1, 5]`
- No per-step re-encoding. No encoder arg in `decoder.rollout()`
- Changes 1+2 already provide spatial awareness in the decoder

**Net: +5 params from Changes 1+2. Full model: ~391 → ~396.**

---

## Architecture Summary

```
5D Features [d/C, dist_depot, x, y, angle/π]   (STATIC — computed once)
    → Linear(5→2) + L2Norm → ψ on unit circle               (QAP mode)
    → Rotation(MLP(5→16→1, tanh)→θ→R(θ)·ψ) → ψ'            (QAP mode)
    → psi_prime FIXED for all decode steps
    → Context: [ψ'_curr, cap/C, t/N, x_curr, y_curr] → Wq(6→2) → query  ← Change 2
    → Scoring: q·ψ'_j + λ·E_kNN(j) − μ·dist(vₜ,vⱼ) → softmax → action  ← Change 1
    → PPO training (ε=0.2, K=3, γ=0.99, GAE λ=0.95)
```

### Encoder (2 variants)
**QAP mode (`encoder_type="qap"`, default):**
1. **FeatureConstructor:** `[d/C, dist_depot, x, y, angle/π]` → [B, N+1, 5]
2. **AmplitudeProjection:** `Linear(5→2) + F.normalize` → [B, N+1, 2] (unit norm!)
3. **RotationMLP:** `MLP(5→16→1, tanh)` → θ_i → [B, N+1]
4. **Rotation:** `R(θ_i) · ψ_i` → ψ'_i → [B, N+1, 2]
5. **Called ONCE** — psi_prime fixed for all decode steps

**Baseline mode (`encoder_type="baseline"`, ablation):**
1. **FeatureConstructor:** same 5D features
2. **BaselineEncoder:** `Linear(5→2) + ReLU` (STATIC — no re-encoding)

### Decoder (autoregressive, N steps)
5. **ContextQuery:** `[ψ'_curr, cap/C, t/N, x_curr, y_curr]` → `Wq(6→2)` → `(query, current_coords)`
6. **HybridScoring:** `q·ψ'_j + λ·E_kNN(j) − μ·dist(vₜ,vⱼ)` → scores [B, N+1]
7. **Masking:** infeasible → -1e9, depot ALWAYS feasible → log_softmax → sample

### Parameter Budget

| Model | Actor | Full |
|-------|-------|------|
| QAP-DRL (Changes 1+2) | ~139 | **~396** |
| Pure DRL baseline | ~25 | ~283 |

---

## Current Hyperparameters (Phase 1b — v9)

```python
BATCH_SIZE    = 512
EPOCH_SIZE    = 128_000
ENTROPY_COEF  = 0.01
KNN_K         = 10
AUG_SAMPLES   = 8
MU_INIT       = 0.5    # Change 1
LAMBDA_INIT   = 0.1
VAL_EVAL_SIZE = 10_000  # key paper standard
ORTOOLS_EVAL_SIZE = 1_000  # OR-Tools subset (speed)
```

---

## File Status (post-changes)

| File | Changes | Status |
|------|---------|--------|
| `encoder/feature_constructor.py` | Static 5D (Change 3 reverted) | ✓ |
| `encoder/amplitude_projection.py` | input_dim=5 (Change 3 reverted) | ✓ |
| `encoder/rotation_mlp.py` | input_dim=5 (Change 3 reverted) | ✓ |
| `encoder/qap_encoder.py` | input_dim=5, static (Change 3 reverted) | ✓ |
| `decoder/context_query.py` | Change 2: ctx ℝ⁴→ℝ⁶ | ✓ |
| `decoder/hybrid_scoring.py` | Change 1: mu_param, −μ·dist | ✓ |
| `decoder/qap_decoder.py` | C1+C2: current_coords to scoring (no encoder arg) | ✓ |
| `models/qap_policy.py` | C1+C2: feature_dim=5, broadcast psi_prime | ✓ |
| `training/ppo_agent.py` | C1: mu_val logged | ✓ |
| `training/evaluate.py` | v3: coord-aug+greedy (Change 3 fix) | ✓ |
| `utils/ortools_solver.py` | v3: +solve_one_with_routes() | ✓ |
| `train_n20.py` | All: mu_val logged, chart updated, ortools_route.png | ✓ |
| `train_n50.py` | OR-Tools route map, twinx gridline fix | ✓ |

---

## Implementation Rules

1. PyTorch only — NO PennyLane, Qiskit
2. Batch-first: `[batch_size, N+1, ...]`
3. QAP mode: ALWAYS L2-normalize after amplitude projection
4. Mask BEFORE softmax (-1e9), NEVER after
5. Depot (index 0) NEVER masked
6. kNN precomputed once per instance from spatial coords (not per step)
7. Shared encoder between actor and critic
8. psi_prime DETACHED before critic head
9. demands depot = 0
10. Feature order: [d/C, dist_depot, x, y, angle/π]  (5D — Change 3 reverted)
11. Angle normalized by π (range [-1, 1])
12. ψ'_curr = zero vector at depot; x_curr,y_curr = actual depot coords (NOT zero)
13. `context_query.forward()` returns TUPLE — always unpack: `query, current_coords = ...`
14. `decoder.rollout()` uses fixed psi_prime — no encoder arg, no per-step re-encoding
15. `evaluate_actions()` broadcasts static psi_prime — do NOT rebuild per step
16. Run train scripts from inside cvrp-ppo/

---

## Validation Checks

After any code change, verify:
- [ ] `features.shape[-1] == 5` — 5D features (Change 3 reverted)
- [ ] ψ' vectors: L2 norm = 1.0 (atol=1e-5) — QAP mode only
- [ ] All tours feasible, all N customers visited exactly once
- [ ] `context.shape[-1] == 6` — 6D context after Change 2
- [ ] `current_coords.shape == (B, 2)` — returned by context_query
- [ ] `dist_to_nodes.shape == (B, N+1)` — Change 1
- [ ] `hasattr(model.decoder.hybrid, 'mu_param')` — Change 1
- [ ] `mask[:, 0].sum() == 0` — depot never masked
- [ ] clip_fraction 2-15%
- [ ] adv_std > 0.8 early
- [ ] entropy 0.5-1.0

---

## Pitfalls Reference

| # | Pitfall | Fix |
|---|---------|-----|
| P1 | Missing L2 norm after projection | `F.normalize(p=2, dim=-1)` |
| P2 | Mask after softmax | -1e9 BEFORE log_softmax |
| P3 | Depot masked | `mask[:, 0] = False` |
| P4 | kNN self-loops | `diagonal.fill_(inf)` before topk |
| P5 | NaN in rotation | `torch.clamp(theta, -10, 10)` |
| P13 | Wrong device | `.to(device)` everywhere |
| P14 | OOM RTX 3050 | batch_size=256; empty_cache() |
| P15 | context_query returns tuple | `query, current_coords = context_query(...)` |
| P16 | mu_param missing | `nn.Parameter(torch.tensor(0.5))` |
| P17 | Change 3 re-applied (dynamic 6D encoder) | REVERTED. Encoder must be STATIC 5D |
| P18 | Per-step re-encoding attempted | NEVER re-encode. psi_prime fixed after initial encode |
| P19 | evaluate_actions rebuilds psi_prime per step | Must BROADCAST static psi_prime |
| P20 | feature_dim set to 6 | Must be 5. Change 3 reverted |
| P21 | evaluate_augmented uses stochastic sampling | coord aug + greedy (evaluate.py v3) |
| P22 | val_tour > greedy_tour in log | Check evaluate.py — stochastic aug broken |
| P23 | λ goes negative or μ > 2 | Likely Change 3 contamination. Verify encoder is static 5D |
| P24 | Dense grey gridlines on twinx panels | `.set_zorder(-1)`, `.patch.set_visible(False)`, `.grid(False)` |
