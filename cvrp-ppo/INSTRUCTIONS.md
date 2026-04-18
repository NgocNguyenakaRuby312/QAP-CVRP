# CVRP-PPO — Developer Instructions & Workflow Guide

> **Purpose of this document:**  
> A complete step-by-step guide for anyone who wants to understand, run, modify, or extend this codebase.  
> It covers the research motivation, software requirements, folder structure, data format, training workflow,  
> evaluation workflow, and how every module connects to the methodology diagram.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Methodology Overview](#2-methodology-overview)
3. [Requirements](#3-requirements)
4. [Folder Structure](#4-folder-structure)
5. [File-by-File Purpose](#5-file-by-file-purpose)
6. [Data Format](#6-data-format)
7. [Configuration System](#7-configuration-system)
8. [Step-by-Step Workflow](#8-step-by-step-workflow)
9. [Training Guide](#9-training-guide)
10. [Evaluation Guide](#10-evaluation-guide)
11. [Testing Guide](#11-testing-guide)
12. [Extending the Codebase](#12-extending-the-codebase)
13. [Troubleshooting](#13-troubleshooting)
14. [Notation Reference](#14-notation-reference)

---

## 1. What This Project Does

This repository implements a **constructive neural solver for the Capacitated Vehicle Routing Problem (CVRP)**  
trained with **Proximal Policy Optimization (PPO)**.

### The Problem
Given:
- A **depot** at coordinate `(x₀, y₀)`
- `N` **customers**, each at `(xᵢ, yᵢ)` with demand `dᵢ`
- A **vehicle capacity** `C`

Find the set of routes — each starting and ending at the depot — that:
- Visits every customer **exactly once**
- Never exceeds capacity `C` on a single route leg
- **Minimizes total travel distance**

### Our Approach
Rather than using the standard Transformer encoder from prior work (Kool et al., 2019),  
this solver introduces a **novel geometric encoding pipeline**:

1. **K-Means clustering** decomposes large instances into tractable sub-problems
2. **5D feature vectors** encode demand, distance, position, and angular context per node
3. **Amplitude projection** maps each node onto the 2D unit circle (`α² + β² = 1`)
4. **Per-node MLP rotation** adaptively rotates each node's embedding in polar space
5. **Hybrid EkNN + dot-product scoring** selects the next node at each decoder step
6. **PPO** with a clipped surrogate objective trains the full end-to-end policy

---

## 2. Methodology Overview

The full pipeline (matching the methodology diagram):

```
┌──────────────────────────────────────────────────────────────────┐
│                         CVRP INPUT                               │
│            Depot (x₀,y₀)  +  Customers {(xᵢ,yᵢ,dᵢ)}  +  C     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
           ╔════════════════════════════════╗
           ║           ENCODER              ║
           ║                                ║
           ║  ① K-Means Clustering          ║
           ║     N customers → K clusters   ║
           ║     O(N²) → O((N/K)²)          ║
           ║                                ║
           ║  ② Feature Construction        ║
           ║     xᵢ = [dᵢ/C, dist,          ║
           ║            x̂, ŷ, angle/π] ∈ ℝ⁵║
           ║                                ║
           ║  ③ Amplitude Projection        ║
           ║     ψᵢ = Normalize(W·xᵢ + b)   ║
           ║     ‖ψᵢ‖ = 1  (unit circle)    ║
           ║                                ║
           ║  ④ Per-Node Rotation           ║
           ║     θᵢ = MLP(xᵢ)              ║
           ║     ψ'ᵢ = R(θᵢ) · ψᵢ           ║
           ╚══════════════════╤═════════════╝
                              │  ψ' [B, N+1, 2]
           ╔══════════════════╧═════════════╗
           ║           DECODER              ║◄────────────┐
           ║                                ║             │
           ║  ⑤ Context → Query             ║             │
           ║     ctx = [ψ'_curr,            ║             │
           ║             cap/C, t/N] ∈ ℝ⁴   ║             │
           ║     query = Wq·ctx ∈ ℝ²        ║             │
           ║                                ║             │
           ║  ⑥ Hybrid Scoring              ║             │
           ║     Score(j) = query·ψ'ⱼ       ║             │
           ║              + λ·EkNN(j)       ║             │
           ║     P(j) = softmax(Score(j))   ║             │
           ║                                ║             │
           ║     All customers served? ─No──┘             │
           ║              │ Yes                            │
           ╚══════════════╧═════════════════╝             │
                          │                               │
           ╔══════════════╧═════════════════╗      Repeat node
           ║         PPO UPDATE   ⑦          ║      selection
           ║                                ║
           ║  L = E[min(rₜ·Aₜ,              ║
           ║         clip(rₜ,1±ε)·Aₜ)]     ║
           ║  R = −Total Distance            ║
           ╚══════════════╤═════════════════╝
                          │
           ┌──────────────┴─────────────────┐
           │        OPTIMIZED ROUTES        │
           │  Depot→C₁→C₂→…→Depot          │
           │  ∀ routes: Σdᵢ ≤ C             │
           └────────────────────────────────┘
```

---

## 3. Requirements

### 3.1 Python Version

```
Python >= 3.9
```

Python 3.9, 3.10, or 3.11 are all tested and supported.  
Python 3.8 is **not** recommended (some f-string and type-hint syntax may fail).

### 3.2 Hardware Requirements

| Mode | Minimum | Recommended |
|---|---|---|
| CPU-only (testing) | 8 GB RAM | 16 GB RAM |
| GPU training (CVRP-50) | 4 GB VRAM | 8 GB VRAM |
| GPU training (CVRP-100) | 8 GB VRAM | 16 GB VRAM |
| GPU training (CVRP-200+) | 16 GB VRAM | 24 GB VRAM |

A CUDA-capable NVIDIA GPU is **strongly recommended** for training.  
Inference and testing can be run on CPU.

### 3.3 Python Dependencies

All dependencies are listed in `requirements.txt`:

| Package | Version | Purpose |
|---|---|---|
| `torch` | `>= 2.0.0` | Core deep learning framework. All tensors, autograd, and neural network modules are built with PyTorch. |
| `numpy` | `>= 1.24.0` | Numerical utilities used internally by scikit-learn and for data serialisation. |
| `scikit-learn` | `>= 1.2.0` | Provides `sklearn.cluster.KMeans` for Step 1 (K-Means decomposition). Used only during data generation, not during GPU training. |
| `matplotlib` | `>= 3.7.0` | Route visualisation in `utils/metrics.py` (plot routes, cluster assignments, training curves). |
| `pyyaml` | `>= 6.0` | Parses `configs/*.yaml` experiment configuration files. |
| `tqdm` | `>= 4.65.0` | Progress bars in long-running scripts (evaluate, solve). |
| `wandb` | `>= 0.15.0` | *Optional.* Weights & Biases experiment tracking. Set `use_wandb: false` in config to disable entirely. |
| `tensorboard` | `>= 2.13.0` | *Optional.* TensorBoard logging. Set `use_tensorboard: false` in config to disable. |

### 3.4 Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/cvrp-ppo
cd cvrp-ppo

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

### 3.5 CUDA Setup (Optional but Recommended)

```bash
# Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

If CUDA is not available, set `device: cpu` in your config file.  
The codebase handles the CPU/GPU fallback automatically in `scripts/train.py`.

---

## 4. Folder Structure

```
cvrp-ppo/
│
├── README.md                    ← Project overview and quick-start
├── INSTRUCTIONS.md              ← THIS FILE — full developer guide
├── requirements.txt             ← Python package dependencies
├── .gitignore                   ← Excluded files (outputs, checkpoints, etc.)
│
├── data/                        ← Instance generation (Step 1)
│   └── generator.py             ← CVRPGenerator + K-Means + CVRPDataset
│
├── environment/                 ← CVRP Markov Decision Process
│   └── cvrp_env.py              ← CVRPEnv: reset, step, reward, mask
│
├── encoder/                     ← Encoder pipeline (Steps 2, 3, 4)
│   ├── __init__.py              ← CVRPEncoder: chains Steps 2→3→4
│   ├── feature_construction.py  ← Step 2: builds 5D node features
│   ├── amplitude_projection.py  ← Step 3: linear → unit-norm 2D vectors
│   └── rotation.py              ← Step 4: MLP rotation per node
│
├── decoder/                     ← Decoder pipeline (Steps 5, 6)
│   ├── __init__.py              ← CVRPDecoder: chains Steps 5→6
│   ├── context.py               ← Step 5: context vector → query
│   └── hybrid_scoring.py        ← Step 6: EkNN + dot-product scoring
│
├── models/                      ← Full policy + critic assembly
│   └── policy.py                ← CVRPPolicy (actor) + CVRPCritic (value net)
│
├── training/                    ← PPO training loop (Step 7)
│   ├── ppo_trainer.py           ← PPOTrainer: rollout → update loop
│   └── rollout_buffer.py        ← Trajectory storage + GAE computation
│
├── inference/                   ← (Extend here) decoding strategies
│   └── (greedy.py, sampling.py, augmentation.py — add as needed)
│
├── utils/                       ← Shared utilities
│   └── metrics.py               ← Distance, gap%, Logger, checkpoint save/load
│
├── configs/                     ← YAML experiment configurations
│   └── cvrp100.yaml             ← Default config: CVRP-100, K=5 clusters
│
├── scripts/                     ← Entry points
│   ├── train.py                 ← Main training script
│   └── evaluate.py              ← Evaluation on test set
│
└── tests/                       ← Unit tests
    └── test_shapes.py           ← Shape assertions for every module
```

---

## 5. File-by-File Purpose

This section explains **why** each file exists and **what problem it solves**.

---

### `data/generator.py`

**Purpose:** Creates batches of random CVRP instances and runs K-Means decomposition (Step 1).

**Why it exists:**  
Neural CVRP solvers train on randomly generated instances rather than fixed datasets,  
because the goal is to learn a generalizable policy. This file is the source of all training data.

**Key classes and functions:**

| Name | What it does |
|---|---|
| `kmeans_cluster(coords, n_clusters, seed)` | Wraps `sklearn.KMeans`. Takes `[N, 2]` customer coordinates and returns cluster labels `[N]` and centroids `[K, 2]`. Called once per instance during generation. |
| `CVRPGenerator.__init__` | Stores hyperparameters: `num_loc` (N), `capacity` (C), `max_demand`, `num_clusters` (K). |
| `CVRPGenerator.generate(batch_size, device)` | Samples random coordinates and demands, normalises demands as `dᵢ/max_demand`, runs K-Means if `num_clusters` is set, and returns a dict of tensors. |
| `CVRPGenerator.build_dataset(n_samples)` | Generates a static validation/test set and wraps it in `CVRPDataset`. |
| `CVRPDataset` | PyTorch `Dataset` wrapper. Enables use with `DataLoader` for validation. |

**Output dict structure:**

```python
{
  "coords":          torch.Tensor [B, N+1, 2],   # depot = row 0
  "demands":         torch.Tensor [B, N+1],       # 0 for depot
  "capacity":        torch.Tensor [B],            # scalar C
  "cluster_labels":  torch.Tensor [B, N],         # only if num_clusters set
  "cluster_centers": torch.Tensor [B, K, 2],      # only if num_clusters set
}
```

---

### `environment/cvrp_env.py`

**Purpose:** Implements the CVRP as a Markov Decision Process (MDP) — the interface between the policy and the problem.

**Why it exists:**  
Reinforcement learning requires an environment that accepts actions and returns next states and rewards.  
This file defines that environment for CVRP without any deep learning — it is pure tensor logic.

**Key methods:**

| Method | What it does |
|---|---|
| `reset(instance)` | Takes a generated instance dict, initialises all state tensors (vehicle at depot, capacity full, no customers visited), and returns the initial state dict. |
| `step(state, actions)` | Applies a batch of node-selection actions. Decrements remaining capacity, marks the node as visited, updates the current position, and recomputes the action mask. |
| `get_reward(state, actions)` | After an episode ends, computes `R = −total_route_length` for each instance. This is the PPO training signal. Negative because PPO maximises reward = we minimise distance. |
| `_compute_mask(state)` | Returns a boolean tensor `[B, N+1]` where `True` means the node is a valid next action. A customer is masked if already visited or if its demand exceeds remaining capacity. The depot is always unmasked. |

**State dict keys (maintained throughout an episode):**

| Key | Shape | Meaning |
|---|---|---|
| `coords` | `[B, N+1, 2]` | Node coordinates (unchanging) |
| `demands` | `[B, N+1]` | Node demands (unchanging) |
| `capacity` | `[B]` | Vehicle capacity C (unchanging) |
| `current_node` | `[B]` | Index of last visited node |
| `used_capacity` | `[B]` | Demand consumed on current route leg |
| `visited` | `[B, N+1]` | Bool mask of served customers |
| `action_mask` | `[B, N+1]` | Bool mask of valid next actions |
| `step_count` | `[B]` | Decoder step counter t |
| `done` | `[B]` | True when all customers served |

---

### `encoder/feature_construction.py`

**Purpose:** Implements **Step 2** — builds the 5-dimensional node feature vector.

**Why it exists:**  
The raw input (x, y coordinates and demand) is not directly useful for a 2D projection.  
This module normalises and enriches the representation so that the downstream linear  
projection (Step 3) receives meaningful, scale-invariant inputs.

**Feature vector definition:**

| Index | Feature | Formula | Range | Why it matters |
|---|---|---|---|---|
| 0 | Normalised demand | `dᵢ / max_demand` | [0, 1] | Encodes how much capacity this customer consumes |
| 1 | Distance to depot | `‖(xᵢ,yᵢ) − (x₀,y₀)‖ / max_dist` | [0, 1] | Encodes spatial remoteness from the depot |
| 2 | Normalised x | `(xᵢ − x_min) / (x_max − x_min)` | [0, 1] | Absolute position (scale-invariant) |
| 3 | Normalised y | `(yᵢ − y_min) / (y_max − y_min)` | [0, 1] | Absolute position (scale-invariant) |
| 4 | Polar angle | `atan2(yᵢ−y₀, xᵢ−x₀) / π` | [-1, 1] | Angular position relative to depot |

The depot row is set to all zeros (depot has no demand, distance 0, angle undefined).

---

### `encoder/amplitude_projection.py`

**Purpose:** Implements **Step 3** — maps each 5D feature vector onto the 2D unit circle.

**Why it exists:**  
This is the core representational innovation of the architecture. By projecting all nodes  
onto the unit circle (`α² + β² = 1`), the model creates a shared polar coordinate space  
where every node's position is defined purely by its angle. This enables:
- The per-node rotation (Step 4) to have a geometrically meaningful effect
- The EkNN scoring (Step 6) to compute dot-products that measure angular similarity
- Visualisation of all node embeddings on a 2D circle

**How it works:**
```
xᵢ ∈ ℝ⁵  →  W·xᵢ + b ∈ ℝ²  →  Normalize(·)  →  ψᵢ = [αᵢ, βᵢ] ∈ ℝ²  with ‖ψᵢ‖=1
```

**Important:** The unit-norm constraint is **not a regulariser** — it is a hard geometric  
constraint enforced by L2 normalisation. Gradients flow through `F.normalize()` correctly.

---

### `encoder/rotation.py`

**Purpose:** Implements **Step 4** — adaptively rotates each node's unit-norm embedding.

**Why it exists:**  
A fixed projection (Step 3) maps all nodes to the unit circle but doesn't account for  
the structural role of each node in the routing solution. The rotation MLP allows the  
encoder to learn **node-specific angular adjustments** — effectively reorganising where  
each node sits on the unit circle based on its features, making nodes that tend to be  
visited consecutively cluster together in embedding space.

**How it works:**
```
θᵢ = MLP(xᵢ)               θᵢ ∈ [−π, π]  (bounded by π·tanh)
ψ'ᵢ = R(θᵢ) · ψᵢ

       [cos θᵢ  −sin θᵢ]
R(θᵢ) =                  
       [sin θᵢ   cos θᵢ]
```

**Unit norm is preserved** because rotation matrices are orthogonal (`Rᵀ R = I`),  
so `‖R·ψ‖ = ‖ψ‖ = 1` always holds.

**Key classes:**

| Class | Purpose |
|---|---|
| `RotationMLP` | The MLP `ℝ⁵ → ℝ` predicting θᵢ. Output bounded via `π·tanh`. |
| `rotation_matrix_2d(theta)` | Constructs batched `[B, N+1, 2, 2]` rotation matrices from angle tensor. |
| `PerNodeRotation` | Combines MLP + rotation application. Returns `(ψ', θ)`. |

---

### `encoder/__init__.py`

**Purpose:** Chains Steps 2 → 3 → 4 into a single `CVRPEncoder` module.

**Why it exists:**  
The policy needs to call the encoder once at the start of each episode to produce  
node embeddings `ψ'` that are reused at every decoder step. This module provides  
that single clean interface.

**Forward pass:**
```
state dict  →  FeatureBuilder  →  features [B, N+1, 5]
            →  AmplitudeProjection  →  ψ [B, N+1, 2]
            →  PerNodeRotation  →  ψ' [B, N+1, 2]  +  θ [B, N+1]
```

---

### `decoder/context.py`

**Purpose:** Implements **Step 5** — assembles the context vector from the current decoding state and projects it to a query.

**Why it exists:**  
The decoder needs to know the current state of the routing problem at each step:  
where the vehicle is, how much capacity is remaining, and how far through the episode  
we are. This information is encoded into a 4D context vector and projected to the  
same 2D space as the node embeddings so dot-products can be computed in Step 6.

**Context vector components:**

| Component | Dimension | Value | Purpose |
|---|---|---|---|
| `ψ'_curr[0]` | 1 | α of current node embedding | Encodes the position of the current node on unit circle |
| `ψ'_curr[1]` | 1 | β of current node embedding | Encodes the position of the current node on unit circle |
| `cap/C` | 1 | `(C − used) / C` ∈ [0,1] | How full is the vehicle? |
| `t/N` | 1 | `step / N` ∈ [0,1] | How far through the episode are we? |

Query: `query = Wq · ctx  ∈ ℝ²`  (same space as `ψ'` → dot-products are meaningful)

---

### `decoder/hybrid_scoring.py`

**Purpose:** Implements **Step 6** — computes per-node scores combining attention and EkNN, then selects the next node.

**Why it exists:**  
Standard attention-based decoders use only a dot-product between the query and each  
node embedding. The EkNN term adds a **local neighbourhood awareness** signal: nodes  
that are surrounded by many similar embeddings (likely to be in the same spatial cluster)  
receive a bonus. This encourages the decoder to build locally coherent routes.

**Score formula:**
```
Score(j) = query · ψ'ⱼ   +   λ · EkNN(j)
              ↑ global           ↑ local
         attention            neighbourhood

EkNN(j) = Σ_{j' ∈ kNN(j)}  ψ'ⱼ · ψ'ⱼ'
```

**λ is a learnable scalar** (stored as `log_lambda`, exponentiated to enforce `λ ≥ 0`).  
It will self-tune during PPO training to find the right balance between global attention  
and local neighbourhood cohesion.

**Masking:** Infeasible nodes (visited or over-capacity) are set to `−∞` before softmax,  
giving them exactly zero selection probability.

---

### `decoder/__init__.py`

**Purpose:** Chains Steps 5 → 6 into a single `CVRPDecoder` module.

**One step forward:**
```
state + ψ'  →  ContextAndQuery  →  query [B, 2]
            →  HybridScoring   →  logits [B, N+1], probs [B, N+1], mask [B, N+1]
```

---

### `models/policy.py`

**Purpose:** Full actor (`CVRPPolicy`) and value network (`CVRPCritic`) for PPO.

**Why it exists:**  
PPO requires two networks: the **actor** (policy) that selects actions, and the **critic**  
(value network) that estimates state values for advantage computation. Both networks  
share the encoder architecture but have separate weights to avoid gradient conflicts.

**CVRPPolicy — actor:**
- Calls `CVRPEncoder` once at episode start → `ψ'` cached for all decoder steps
- Loops: calls `CVRPDecoder` → selects node → calls `env.step` → until `done`
- `evaluate_actions(state, env, actions)` re-evaluates stored actions under current policy — this is the function called during PPO updates to compute `π_new(a|s)`

**CVRPCritic — value network:**
- Independent `CVRPEncoder` (separate weights from actor encoder)
- Mean-pools `ψ'` across all nodes → `h ∈ ℝ²`
- Concatenates `h` with `[cap/C, t/N]` → 4D input to MLP head → scalar `V(s)`
- Used to compute advantages `Aₜ = Rₜ − V(sₜ)` during rollout

---

### `training/rollout_buffer.py`

**Purpose:** Stores per-step experience tuples during rollout and computes Generalised Advantage Estimates (GAE).

**Why it exists:**  
PPO collects a full episode of experience, then performs multiple gradient updates  
on it before discarding. The buffer stores `(log_prob, value, action)` per step,  
and after the episode computes advantages using GAE — a weighted combination of  
temporal difference errors that balances bias vs. variance in the advantage estimate.

**GAE formula:**
```
δₜ  = rₜ + γ·V(sₜ₊₁) − V(sₜ)          (TD error)
Aₜ  = Σ_{l≥0} (γλ)^l · δₜ₊ₗ           (GAE)
```

**Key methods:**

| Method | When to call | What it does |
|---|---|---|
| `clear()` | Start of each iteration | Resets all lists |
| `add_step(log_prob, value, action)` | Each decoder step | Appends one step of experience |
| `set_rewards(rewards)` | After episode ends | Sets terminal reward (−total_dist) |
| `compute_gae(last_value, gamma, gae_lambda)` | After `set_rewards` | Fills `advantages` and `returns` tensors |
| `get_minibatches(n_minibatches)` | During PPO update | Yields shuffled minibatches for gradient steps |

---

### `training/ppo_trainer.py`

**Purpose:** Implements **Step 7** — the full PPO training loop.

**Why it exists:**  
This is the main training engine. It orchestrates: collecting rollouts, computing  
advantages, performing multiple gradient updates with the clipped PPO objective,  
logging, and checkpointing.

**PPO clipped objective:**
```
rₜ       = exp(log π_new(aₜ|sₜ) − log π_old(aₜ|sₜ))   ← importance sampling ratio
L_CLIP   = E[ min(rₜ·Aₜ,  clip(rₜ, 1−ε, 1+ε)·Aₜ) ]
L_VALUE  = E[ (V(sₜ) − Rₜ)² ]
L_ENTROPY = −H[π(·|sₜ)]                                  ← exploration bonus
L_TOTAL  = −L_CLIP + c₁·L_VALUE − c₂·L_ENTROPY
```

**Training iteration (one call to `train()` loop body):**
```
1. collect_rollout()      ← run policy in env for B instances, store experience
2. buffer.set_rewards()   ← assign R = −total_distance
3. buffer.compute_gae()   ← compute Aₜ and Rₜ for all steps
4. for _ in ppo_epochs:
     for minibatch in buffer.get_minibatches():
       evaluate_actions()   ← re-compute log π_new for stored actions
       compute_policy_loss()
       compute_value_loss()
       compute_entropy()
       total_loss.backward()
       clip_grad_norm()
       optimizer.step()
5. log metrics
6. (periodically) validate, save checkpoint
```

---

### `utils/metrics.py`

**Purpose:** Contains three utilities: solution quality metrics, training logger, and checkpoint management.

**Why it's one file:** All three are small utilities with no external dependencies beyond each other.

**Metrics functions:**

| Function | Purpose |
|---|---|
| `compute_total_distance(state, actions)` | Sums Euclidean edge lengths for all routes including depot returns. Returns `[B]` tensor. |
| `optimality_gap(sol_dist, opt_dist)` | Computes `100 × (sol − opt) / opt` percentage gap vs. a reference solver. |
| `compute_metrics(state, actions, opt_dist)` | Convenience wrapper returning a dict with `mean_dist`, `std_dist`, and optionally `mean_gap`. |

**Logger class:**
- Writes JSON lines to `train_log.jsonl` in the log directory (always)
- Optionally forwards to Weights & Biases (if `use_wandb: true` in config)
- Optionally forwards to TensorBoard (if `use_tensorboard: true` in config)
- Call `logger.log_scalars(metrics_dict, step)` to log any scalar dict

**Checkpoint functions:**

| Function | Purpose |
|---|---|
| `save_checkpoint(policy, critic, optimizer, iteration, metrics, path)` | Saves full training state (weights + optimizer states + iteration number). Safe to resume from. |
| `load_checkpoint(path, policy, critic, optimizer)` | Restores training state. Returns the saved iteration number so training continues from the right step. |
| `save_best_model(policy, metric_val, best_val, path)` | Saves policy weights only if `metric_val > best_val`. Used to track the best validation checkpoint. |

---

### `configs/cvrp100.yaml`

**Purpose:** Central configuration file for CVRP-100 experiments.

**Why YAML configs exist:**  
Separating hyperparameters from code means you can run different experiments  
(different N, K, λ, learning rate, etc.) without editing Python files. Every  
parameter used in `scripts/train.py` comes from this file.

**Config sections:**

| Section | Controls |
|---|---|
| `data` | N, C, max_demand, K (Step 1), validation set size |
| `encoder` | Feature dimension (5), rotation MLP hidden size |
| `decoder` | EkNN k, initial λ, whether λ is learnable |
| `ppo` | All PPO hyperparameters: ε, c₁, c₂, epochs, minibatches, γ, λ_GAE |
| `training` | Iterations, batch size, learning rate, eval/checkpoint frequency, device, output directory |
| `logging` | W&B / TensorBoard toggle, project name, run name |

---

### `scripts/train.py`

**Purpose:** Main entry point for training. Reads config, builds all components, starts the trainer.

**Usage:**
```bash
python scripts/train.py --config configs/cvrp100.yaml
python scripts/train.py --config configs/cvrp100.yaml --device cuda
python scripts/train.py --config configs/cvrp100.yaml --resume outputs/cvrp100/ckpt_0050.pt
```

---

### `scripts/evaluate.py`

**Purpose:** Loads a checkpoint and evaluates it with greedy + augmentation decoding on a test set.

**Usage:**
```bash
python scripts/evaluate.py \
  --checkpoint outputs/cvrp100/ckpt_0500.pt \
  --config configs/cvrp100.yaml \
  --n_samples 1000 \
  --n_aug 8
```

`n_aug=8` runs 8 coordinate augmentations (4 rotations × 2 reflections) per instance  
and takes the best solution — standard practice from POMO to boost solution quality at inference time.

---

### `tests/test_shapes.py`

**Purpose:** Shape sanity checks for every module. Ensures no silent tensor shape bugs.

**What is tested:**

| Test | Assertion |
|---|---|
| `test_feature_builder` | Output `[B, N+1, 5]`; depot row is all zeros |
| `test_amplitude_projection` | Output `[B, N+1, 2]`; all rows are unit-norm |
| `test_per_node_rotation` | Output `[B, N+1, 2]`; unit norm preserved after rotation |
| `test_encoder` | Full encoder output shapes correct |
| `test_context_query` | Query shape `[B, 2]` |
| `test_hybrid_scoring` | Logits `[B, N+1]`; probs sum to 1 |
| `test_decoder` | All decoder output shapes correct |
| `test_env_step` | Step counter increments; mask shape correct |

**Run tests:**
```bash
pytest tests/test_shapes.py -v
```

---

## 6. Data Format

### Raw Input (CSV)

```
node_id, x,      y,      demand, capacity
0,       0.500,  0.500,  0.000,  1.0        ← depot (always row 0, demand=0)
1,       0.412,  0.773,  0.300,  1.0
2,       0.187,  0.234,  0.150,  1.0
...
```

### Tensor Format (inside the model)

```python
instance = {
  "coords":   tensor [B, N+1, 2],  # row 0 = depot
  "demands":  tensor [B, N+1],     # row 0 = 0.0, rows 1…N = dᵢ/max_demand
  "capacity": tensor [B],          # scalar C, same for all rows
}
```

### Rules

- **Node 0 is always the depot.** The depot must be first with `demand = 0`.
- **All coordinates must be in [0, 1].** The `FeatureBuilder` performs per-instance min-max normalisation automatically, but starting coordinates should be bounded.
- **Demands must satisfy `dᵢ ≤ C` for each customer.** An instance where any single customer exceeds capacity is infeasible by definition.
- **Demands are normalised by `max_demand`** (not by `C`) in the generator. Feature 0 divides by `C` again during feature construction.

---

## 7. Configuration System

All experiments are controlled by YAML files in `configs/`. The `scripts/train.py` script  
reads the config and passes values to constructors — no hardcoded hyperparameters exist in  
the model code itself.

### Creating a new config (example: CVRP-50)

```yaml
# configs/cvrp50.yaml
data:
  num_loc:      50
  capacity:     1.0
  max_demand:   9
  num_clusters: 3        # fewer clusters for smaller instances
  val_size:     1000

encoder:
  feature_dim: 5
  hidden_dim:  64

decoder:
  k:            3        # smaller k for smaller instances
  lambda_init:  0.5
  learn_lambda: true

ppo:
  clip_epsilon:  0.2
  entropy_coef:  0.01
  value_coef:    0.5
  max_grad_norm: 1.0
  ppo_epochs:    3
  n_minibatches: 8
  gamma:         0.99
  gae_lambda:    0.95

training:
  n_iterations:     300
  batch_size:       128   # smaller instances → larger batch
  lr:               1e-4
  eval_every:       10
  checkpoint_every: 50
  device:           cuda
  log_dir:          outputs/cvrp50

logging:
  use_wandb:       false
  use_tensorboard: false
```

Then run:
```bash
python scripts/train.py --config configs/cvrp50.yaml
```

---

## 8. Step-by-Step Workflow

This section walks through exactly what happens when you call `python scripts/train.py`.

### Step 1 — Config is loaded and components are built

```python
gen    = CVRPGenerator(num_loc=100, capacity=1.0, num_clusters=5)
env    = CVRPEnv(num_loc=100)
policy = CVRPPolicy(feature_dim=5, hidden_dim=64, k=5)
critic = CVRPCritic(feature_dim=5, hidden_dim=64)
trainer = PPOTrainer(policy, critic, env, gen, ...)
```

### Step 2 — Each PPO iteration begins with rollout collection

```python
instance = gen.generate(batch_size=64)
# → coords [64, 101, 2], demands [64, 101], capacity [64]
# → cluster_labels [64, 100], cluster_centers [64, 5, 2]  (Step 1)

state = env.reset(instance)
# → initialises all state tensors (vehicle at depot, full capacity)
```

### Step 3 — Encoder runs once per episode

```python
psi_prime, features, theta = policy.encoder(state)
# Step 2: features [64, 101, 5]    ← [d/C, dist, x, y, angle/π]
# Step 3: psi      [64, 101, 2]    ← unit-norm projections
# Step 4: psi_prime [64, 101, 2]   ← after per-node MLP rotation
```

### Step 4 — Decoder loops until all customers served

```python
while not state["done"].all():
    logits, probs, mask = policy.decoder(state, psi_prime, step, n_customers)
    # Step 5: query = Wq · [ψ'_curr, cap/C, t/N]
    # Step 6: Score(j) = query·ψ'ⱼ + λ·EkNN(j)
    
    action = torch.multinomial(probs, 1).squeeze(1)  # stochastic sampling
    state  = env.step(state, action)
    step  += 1
```

### Step 5 — Episode ends, reward computed

```python
reward = env.get_reward(state, actions)   # R = −total_distance [64]
buffer.set_rewards(reward)
buffer.compute_gae(last_value=zeros, gamma=0.99, gae_lambda=0.95)
```

### Step 6 — PPO update (Step 7 in the diagram)

```python
for _ in range(ppo_epochs=3):
    for log_p_old, acts, advantages, returns, values_old in buffer.get_minibatches(8):
        log_p_new, entropy = policy.evaluate_actions(init_state, env, acts)
        
        ratio  = (log_p_new - log_p_old).exp()          # rₜ = π_new/π_old
        L_CLIP = -min(ratio * A, clip(ratio, 1±ε) * A).mean()
        L_VF   = (V_new - returns).pow(2).mean()
        L_ENT  = -entropy.mean()
        
        loss   = L_CLIP + 0.5*L_VF + 0.01*L_ENT
        loss.backward()
        clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
```

### Step 7 — Logging and checkpointing

```python
logger.log_scalars({"reward": mean_reward, "policy_loss": ..., ...}, step=it)
# Saved to: outputs/cvrp100/train_log.jsonl

if (it+1) % checkpoint_every == 0:
    save_checkpoint(policy, critic, optimizer, it+1, metrics, "outputs/ckpt_XXXX.pt")
```

---

## 9. Training Guide

### Basic training

```bash
python scripts/train.py --config configs/cvrp100.yaml
```

### Resume from checkpoint

```bash
python scripts/train.py --config configs/cvrp100.yaml \
  --resume outputs/cvrp100/ckpt_0200.pt
```

### Override device

```bash
python scripts/train.py --config configs/cvrp100.yaml --device cpu
```

### Monitor training

```bash
# The trainer prints one line per iteration:
# [0050/500] reward=-15.4231  p_loss=0.0234  v_loss=0.1823  ent=1.4521  (2.3s)

# View full log:
cat outputs/cvrp100/train_log.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"step={d['step']}  reward={d['reward']:.4f}\")"
```

### Expected training progress (CVRP-100)

| Iteration | Expected Mean Reward | Notes |
|---|---|---|
| 0–50 | −30 to −25 | Random policy, learning starts |
| 50–150 | −25 to −20 | Policy discovers basic structure |
| 150–300 | −20 to −17 | Refines routing patterns |
| 300–500 | −17 to −15 | Fine-tuning convergence |

Negative reward = negative total distance. Higher (less negative) is better.

---

## 10. Evaluation Guide

### Greedy decoding (fast)

```bash
python scripts/evaluate.py \
  --checkpoint outputs/cvrp100/ckpt_0500.pt \
  --config configs/cvrp100.yaml \
  --n_samples 1000 \
  --n_aug 1
```

### With 8× augmentation (best quality)

```bash
python scripts/evaluate.py \
  --checkpoint outputs/cvrp100/ckpt_0500.pt \
  --config configs/cvrp100.yaml \
  --n_samples 1000 \
  --n_aug 8
```

### Output

```
[Eval] n=1000  aug=8
  Mean total distance : 15.4231
  Std  total distance : 0.8712
```

---

## 11. Testing Guide

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_shapes.py::test_amplitude_projection -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

All tests should pass on both CPU and GPU. Tests use small instances (`B=4, N=20`)  
so they complete in under 5 seconds.

---

## 12. Extending the Codebase

### Add a new problem variant (e.g., CVRP with time windows)

1. Add a new generator in `data/generator.py` or create `data/cvrptw_generator.py`
2. Add time-window fields to the state dict in `environment/cvrp_env.py`
3. Add time-window penalty to `get_reward()`
4. Add time-window features (e.g., `earliest`, `latest`, `service_time`) to `encoder/feature_construction.py` — change `feature_dim` from 5 to 7 or 8
5. Update `configs/cvrptw100.yaml` with the new `feature_dim`

### Change the encoder (e.g., replace rotation with multi-head attention)

1. Create `encoder/attention_encoder.py` with a new class implementing the same interface:
   ```python
   def forward(self, state: dict):
       # returns: psi_prime [B, N+1, D], features [B, N+1, 5], extras
   ```
2. Update `encoder/__init__.py` to import and use the new encoder
3. Update `embed_dim` in `decoder/context.py` and `decoder/hybrid_scoring.py` to match the new output dimension `D`

### Add a new decoding strategy (e.g., beam search)

1. Create `inference/beam_search.py`
2. Import and call it from `scripts/evaluate.py` by adding a `--decode_type` argument

### Change the training algorithm (e.g., use REINFORCE instead of PPO)

1. Create `training/reinforce_trainer.py`
2. Replace `PPOTrainer` with `REINFORCETrainer` in `scripts/train.py`
3. The policy and environment interfaces do not need to change

---

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'sklearn'`

```bash
pip install scikit-learn
```

### `CUDA out of memory`

Reduce `batch_size` in the config (try `32` or `16`) or reduce `num_loc`.

### `RuntimeError: Expected all tensors to be on the same device`

Ensure `device` in config matches where your tensors are. Check that `gen.generate(B, device=device)` and `env.reset(instance)` both use the same device string.

### Loss is `nan` after a few iterations

Likely a gradient explosion. Try:
- Lowering `lr` from `1e-4` to `1e-5`  
- Lowering `max_grad_norm` from `1.0` to `0.5`  
- Checking that `demand > 0` for all customers (zero demand can cause division issues in masking)

### K-Means is slow for large batches

K-Means runs on CPU via scikit-learn. For `batch_size=512` with `N=200`, it can take  
several seconds per batch. To speed up, reduce `num_clusters` or pre-generate a fixed  
training dataset with `gen.build_dataset(50000)` and load from disk instead.

### `rewards are all the same` / `policy not improving`

Check that `entropy` in the log is not collapsing to zero — this indicates the policy  
has become deterministic too early. Increase `entropy_coef` from `0.01` to `0.05`.

---

## 14. Notation Reference

| Symbol | Meaning | Shape / Range |
|---|---|---|
| B | Batch size | scalar |
| N | Number of customers | scalar |
| K | Number of K-Means clusters | scalar |
| k | EkNN neighbourhood size | scalar |
| C | Vehicle capacity | scalar |
| xᵢ | 5D feature vector for node i | `ℝ⁵` |
| ψᵢ | Unit-norm amplitude vector (Step 3 output) | `ℝ²`, `‖ψ‖=1` |
| ψ'ᵢ | Rotated amplitude vector (Step 4 output) | `ℝ²`, `‖ψ'‖=1` |
| θᵢ | Per-node rotation angle | `[−π, π]` |
| R(θᵢ) | 2×2 rotation matrix | `ℝ^{2×2}` |
| ctx | Context vector at decoder step t | `ℝ⁴` |
| query | Projected context (= Wq·ctx) | `ℝ²` |
| λ | EkNN blend weight (learnable) | `ℝ⁺` |
| EkNN(j) | Neighbourhood score for node j | scalar |
| Score(j) | Hybrid score for node j | scalar |
| P(j) | Selection probability for node j | `[0, 1]` |
| R | Episode reward | `−total_dist` |
| Aₜ | GAE advantage at step t | scalar |
| rₜ | PPO importance sampling ratio | `ℝ⁺` |
| ε | PPO clip range | default `0.2` |
| γ | Reward discount | default `0.99` |
| λ_GAE | GAE smoothing parameter | default `0.95` |

---

*Last updated: March 2026*  
*For questions, open an issue on the GitHub repository.*
