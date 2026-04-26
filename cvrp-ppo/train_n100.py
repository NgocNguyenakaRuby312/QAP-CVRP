#!/usr/bin/env python
"""
train_n100.py — Train QAP-DRL on CVRP-100.

    python train_n100.py

N=100 settings: CAPACITY=50, BATCH_SIZE=256, KNN_K=20.
Otherwise identical to train_n20.py (Phase 2: 4D, hidden_dim=32, warm restarts).
"""

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Override settings BEFORE calling train_n20's main
import train_n20

train_n20.GRAPH_SIZE         = 100
train_n20.CAPACITY           = 50       # Kool et al. 2019: N=100 → C=50
train_n20.BATCH_SIZE         = 256      # lower than N=20 (512) — VRAM constraint
train_n20.KNN_K              = 20       # 20% of N=100 graph
train_n20.ORTOOLS_EVAL_SIZE  = 500      # N=100 is slower for OR-Tools
train_n20.ORTOOLS_TIME_LIMIT = 5.0      # N=100 needs more time per instance
train_n20.OUTPUT_DIR         = os.path.join(SCRIPT_DIR, "outputs", "n100")
train_n20.EPOCH_DIR          = os.path.join(train_n20.OUTPUT_DIR, "epochs")
train_n20.ARCHIVE_DIR        = os.path.join(train_n20.OUTPUT_DIR, "Archive")
train_n20.VAL_PATH           = os.path.join(SCRIPT_DIR, "datasets", "val_n100.pkl")
train_n20.BATCHES_PER_EPOCH  = train_n20.EPOCH_SIZE // train_n20.BATCH_SIZE
train_n20.TOTAL_OPT_STEPS    = train_n20.N_EPOCHS * train_n20.BATCHES_PER_EPOCH * 3 * 8

if __name__ == "__main__":
    train_n20.main()
