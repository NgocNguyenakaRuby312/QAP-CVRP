#!/usr/bin/env python
"""
train_n50.py — Train QAP-DRL on CVRP-50.

    python train_n50.py

N=50 settings: CAPACITY=40, BATCH_SIZE=256, KNN_K=15.
Otherwise identical to train_n20.py (Phase 2: 4D, hidden_dim=32, warm restarts).
"""

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Override settings BEFORE calling train_n20's main
import train_n20

train_n20.GRAPH_SIZE         = 50
train_n20.CAPACITY           = 40       # Kool et al. 2019: N=50 → C=40
train_n20.BATCH_SIZE         = 256      # lower than N=20 (512) — VRAM constraint
train_n20.KNN_K              = 15       # 30% of N=50 graph
train_n20.OUTPUT_DIR         = os.path.join(SCRIPT_DIR, "outputs", "n50")
train_n20.EPOCH_DIR          = os.path.join(train_n20.OUTPUT_DIR, "epochs")
train_n20.ARCHIVE_DIR        = os.path.join(train_n20.OUTPUT_DIR, "Archive")
train_n20.VAL_PATH           = os.path.join(SCRIPT_DIR, "datasets", "val_n50.pkl")
train_n20.BATCHES_PER_EPOCH  = train_n20.EPOCH_SIZE // train_n20.BATCH_SIZE
train_n20.TOTAL_OPT_STEPS    = train_n20.N_EPOCHS * train_n20.BATCHES_PER_EPOCH * 3 * 8

if __name__ == "__main__":
    train_n20.main()
