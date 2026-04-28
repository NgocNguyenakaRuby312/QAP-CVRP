#!/usr/bin/env python
"""
generate_all_datasets.py — Regenerate ALL validation and test datasets.

Run this ONCE before training to create fresh fixed datasets:

    python generate_all_datasets.py

This will:
    1. Delete old datasets and OR-Tools cache
    2. Generate new val + test datasets for N=20, 50, 100, 200
    3. Each with 10K instances, fixed seeds for reproducibility

Seeds:
    val:  seed = 12345 (same across all sizes)
    test: seed = 54321 (same across all sizes, different from val)

After running this, you MUST delete the OR-Tools cache so it recomputes:
    - datasets/ortools_refs.json is deleted automatically by this script
"""

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils.data_generator import save_dataset

DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")

# ── Config ───────────────────────────────────────────────────────────────
SIZES = [20, 50, 100, 200]
CAPACITIES = {20: 30, 50: 40, 100: 50, 200: 50}
NUM_SAMPLES = 10_000
VAL_SEED = 12345
TEST_SEED = 54321


def main():
    print("=" * 60)
    print("  Generating ALL datasets (val + test)")
    print("=" * 60)
    print(f"  Sizes      : {SIZES}")
    print(f"  Instances  : {NUM_SAMPLES} per dataset")
    print(f"  Val seed   : {VAL_SEED}")
    print(f"  Test seed  : {TEST_SEED}")
    print(f"  Output     : {DATASETS_DIR}")
    print("=" * 60)

    # Delete stale OR-Tools cache
    ortools_cache = os.path.join(DATASETS_DIR, "ortools_refs.json")
    if os.path.exists(ortools_cache):
        os.remove(ortools_cache)
        print(f"\n  Deleted stale OR-Tools cache: {ortools_cache}")

    os.makedirs(DATASETS_DIR, exist_ok=True)

    for n in SIZES:
        cap = CAPACITIES[n]
        print(f"\n  --- CVRP-{n} (C={cap}) ---")

        # Validation set
        save_dataset(
            graph_size=n,
            num_samples=NUM_SAMPLES,
            capacity=cap,
            path=DATASETS_DIR,
            seed=VAL_SEED,
            filename=f"val_n{n}.pkl",
        )

        # Test set
        save_dataset(
            graph_size=n,
            num_samples=NUM_SAMPLES,
            capacity=cap,
            path=DATASETS_DIR,
            seed=TEST_SEED,
            filename=f"test_n{n}.pkl",
        )

    print("\n" + "=" * 60)
    print("  All datasets generated.")
    print("  OR-Tools cache deleted — will recompute on next training run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
