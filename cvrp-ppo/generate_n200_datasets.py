#!/usr/bin/env python
"""
generate_n200_datasets.py — Generate val and test datasets for CVRP-200.

Run once:  python generate_n200_datasets.py

Creates:
    datasets/val_n200.pkl   (10K instances, seed=12345)
    datasets/test_n200.pkl  (10K instances, seed=54321)
"""

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils.data_generator import save_dataset

DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")

print("=" * 60)
print("  Generating CVRP-200 datasets")
print("=" * 60)

# Validation set — same seed convention as other sizes
save_dataset(
    graph_size=200,
    num_samples=10_000,
    capacity=50,
    path=DATASETS_DIR,
    seed=12345,
    filename="val_n200.pkl",
)

# Test set — different seed
save_dataset(
    graph_size=200,
    num_samples=10_000,
    capacity=50,
    path=DATASETS_DIR,
    seed=54321,
    filename="test_n200.pkl",
)

print("\nDone.")
