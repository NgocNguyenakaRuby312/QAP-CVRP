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
**Two machines:**
- Machine A: `C:\Users\ASUS\Downloads\Thesis Code QAP_VRP\cvrp-ppo\`
- Machine B: `D:\Coding\RubyThesis\QAP-CVRP\cvrp-ppo\`
- **Always `cd` into cvrp-ppo before running — path resolution depends on it**

---

## Architecture Summary

```
5D Features [d/C, dist, x, y, angle/π]
    → Linear(5→2) + L2Norm → ψ on unit circle        (QAP mode)
    → Rotation(MLP(5→16→1, tanh)→θ→R(θ)·ψ) → ψ'     (QAP mode)
    → Autoregressive decoder: context query + hybrid scoring (attention + kNN interference)
    → PPO training (ε=0.2, K=3, γ=0.99, GAE λ=0.95)

Ablation mode (encoder_type="baseline"):
    → Linear(5→2) + ReLU → embedding (no norm, no rotation)
    → Same decoder, same PPO
```

### Encoder (2 variants)
**QAP mode (`encoder_type="qap"`, default):**
1. **FeatureConstructor:** `[d/C, dist, x, y, angle/π]` → [B, N+1, 5]
2. **AmplitudeProjection:** `Linear(5→2) + F.normalize` → [B, N+1, 2] (unit norm!)
3. **RotationMLP:** `MLP(5→16→1, tanh)` → θ_i per node → [B, N+1]
4. **Rotation:** `R(θ_i) · ψ_i` → ψ'_i → [B, N+1, 2] (norm preserved)

**Baseline mode (`encoder_type="baseline"`, ablation):**
1. **FeatureConstructor:** same as above → [B, N+1, 5]
2. **BaselineEncoder:** `Linear(5→2) + ReLU` → [B, N+1, 2] (NOT unit norm)

### Decoder (autoregressive, N steps)
5. **ContextQuery:** `[ψ'_curr, cap_ratio, step_ratio]` → `W_q(4→2, no bias)` → query [B, 2]
6. **HybridScoring:** `q·ψ'_j + λ·Σ_{kNN}(ψ'_i·ψ'_j)` → scores [B, N+1]
7. **Masking:** infeasible → -1e9, depot ALWAYS feasible → log_softmax → sample/greedy

### Critic
- Shared encoder, mean-pool ψ' → MLP(2→64→1) → V(s)
- **psi_prime DETACHED before critic head — prevents value gradient corrupting encoder**

### Parameter Budget

| Model | Actor | Full | Key diff |
|-------|-------|------|----------|
| QAP-DRL (full) | ~134 | ~391 | AmplitudeProj + RotationMLP |
| Pure DRL (baseline) | ~21 | ~278 | Just Linear + ReLU |
| AM (comparison) | ~500K | ~1.4M | Transformer |

---

## Current Hyperparameters (Phase 1b — v8)

```python
# train_n20.py
BATCH_SIZE        = 512       # Phase 1b: was 256 — stronger adv signal
EPOCH_SIZE        = 128_000   # thesis spec
ENTROPY_COEF      = 0.01      # thesis spec — 0.05 caused adv collapse
KNN_K             = 10        # Phase 1: was 5
AUG_SAMPLES       = 8         # inference augmentation
BATCHES_PER_EPOCH = 250       # 128,000 ÷ 512
TOTAL_OPT_STEPS   = 1_200_000 # 200 × 250 × 3 × 8

# ppo_agent.py
eta_min = 1e-5   # v4 fix: was 1e-6 — prevented training freeze
```

---

## Training Diagnostics — What to Watch

| Metric | Healthy | Problem | Action |
|--------|---------|---------|--------|
| clip_fraction | 2–15% | < 0.1% (Phase 1 was 0.014%) | Reduce entropy_coef |
| entropy H[π] | 0.5–1.0 | < 0.5 collapsed / >1.0 too random | Tune entropy_coef |
| adv_std | > 0.8 | < 0.5 | Reduce entropy_coef or increase batch |
| actor loss | stable negative | Rising to 0 | adv collapse — see above |
| LR at ep200 | ≥ 1e-5 | ≈ 0 | eta_min too low (old bug, fixed) |

---

## File Structure (current)

```
cvrp-ppo/
├── run.py
├── options.py
├── train_n20.py              [v8 Phase 1b — active]
├── train_n50.py              [v8 Phase 1b — active]
├── train_n100.py
├── train_n10.py
├── train_ablation_n20.py     [NEW — ablation study]
├── encoder/
│   ├── feature_constructor.py
│   ├── amplitude_projection.py
│   ├── rotation_mlp.py
│   ├── rotation.py
│   ├── qap_encoder.py
│   └── baseline_encoder.py   [NEW — for ablation]
├── decoder/                  [unchanged]
├── environment/              [unchanged]
├── models/
│   └── qap_policy.py         [UPDATED: encoder_type param]
├── training/
│   ├── ppo_agent.py          [v4: eta_min=1e-5]
│   ├── rollout_buffer.py
│   └── evaluate.py           [UPDATED: +evaluate_augmented()]
├── utils/
│   ├── knn.py
│   ├── data_generator.py
│   ├── ortools_refs.py       [UPDATED: rich banner, output_dir param]
│   └── ortools_solver.py     [UPDATED: percentiles, timing stats]
├── datasets/
│   └── ortools_refs.json     [has: mean, std, n_valid — needs re-run for p10-p90]
└── outputs/
    ├── n20/                  [best: 7.228, gap 17.15%, Phase 1 run]
    ├── n50/
    └── ablation_n20/         [NEW]
```

---

## Implementation Rules

1. PyTorch only — NO PennyLane, Qiskit, quantum libraries
2. Batch-first: `[batch_size, N+1, ...]`
3. QAP mode: ALWAYS L2-normalize after amplitude projection
4. Mask BEFORE softmax (set -1e9), NEVER after
5. Depot (index 0) NEVER masked
6. kNN precomputed per instance from spatial coords, no self-loops
7. Shared encoder between actor and critic
8. psi_prime DETACHED before critic head in ppo_agent.update()
9. demands ∈ [1,9] integers; depot demand = 0
10. Feature order: [d/C, dist, x, y, angle/π] (thesis §3.X.3)
11. Angle normalized by π (range [-1, 1])
12. Zero vector for ψ'_curr when at depot (thesis §3.X.6)
13. ALL tensors and models must be on `device`
14. Run train scripts from inside cvrp-ppo/ (not parent directory)

---

## Validation Checks

After any code change, verify:
- [ ] ψ' vectors: L2 norm = 1.0 (atol=1e-5) — QAP mode only
- [ ] All tours feasible (no capacity violations)
- [ ] All N customers visited exactly once
- [ ] Depot at start/end of every sub-route
- [ ] Score shape: [B, N+1] before masking
- [ ] kNN: no self-loops, computed from spatial coords
- [ ] Feature order: [d/C, dist, x, y, angle/π]
- [ ] Angle feature in [-1, 1] range
- [ ] clip_fraction between 2-15% (not near-zero)
- [ ] adv_std > 0.8 in early training
- [ ] entropy between 0.5-1.0
- [ ] PPO loss decreases in early training
- [ ] All tensors on correct device (P13)
- [ ] No OOM errors (P14)

---

## OR-Tools Reference (Pre-Training)

Runs automatically before training via `ensure_ortools_ref()`. Cached in `datasets/ortools_refs.json`.

Current cached values for CVRP-20:
```
mean_tour = 6.1915
std_tour  = 0.8048
5% target = ≤ 6.5011
```

Percentiles (p10-p90) only available after next fresh OR-Tools run (delete entry from JSON or
change n_instances). Current JSON lacks percentile fields.

OR-Tools banner now prints before every training run showing mean, std, CV%, ±2σ range,
5% gap target as absolute tour length, and current best model gap vs OR-Tools.

---

## Gap Reduction Roadmap

| Phase | Key changes | Expected gap | Status |
|-------|-------------|-------------|--------|
| Phase 1 | ENTROPY=0.05, BATCH=256, kNN=10, aug×8 | 17.15% | Done |
| Phase 1b | ENTROPY=0.01, BATCH=512 | ~12-15% | Next run |
| Phase 2 | amp dim 2→4, rotation hidden 16→32 | ~7-10% | After 1b |
| Phase 3 | 400 epochs, CosineWarmRestarts | <5% | After 2 |

---

## Ablation Study

Run `python train_ablation_n20.py` from inside cvrp-ppo/.

Runs both models under identical conditions:
- Same seed (1234), hyperparams, val data, training data
- `encoder_type="qap"` vs `encoder_type="baseline"`

Output: side-by-side comparison chart, per-epoch logs, verdict table.

Three valid outcomes:
1. QAP-DRL better → quantum structure contributes
2. Baseline better → quantum encoding hurts (important finding)
3. No difference → null result (also publishable)

---

## GPU / CUDA Setup

**Hardware:** NVIDIA GeForce RTX 3050, 4GB VRAM, CUDA 13.2

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### VRAM Budget

| Problem | batch_size | VRAM | Status |
|---------|-----------|------|--------|
| CVRP-20 | 512 | ~1.0 GB | ✓ Safe (Phase 1b) |
| CVRP-50 | 512 | ~2.0 GB | ✓ Safe |
| CVRP-100 | 256 | ~2.0 GB | ✓ Safe |
| CVRP-100 | 512 | ~4.0 GB | ⚠ Test first |

### Training Time Estimates

| Problem | Per epoch | 200 epochs |
|---------|----------|------------|
| CVRP-20 (B=512) | ~4–6 min | ~13–20 hrs |
| CVRP-50 (B=512) | ~10–15 min | ~33–50 hrs |
