#!/usr/bin/env python
"""
train_n200.py — Train QAP-DRL on CVRP-200.

    python generate_n200_datasets.py   # first time only
    python train_n200.py

N=200 settings: CAPACITY=50, KNN_K=30, ORTOOLS_TIME=5s.
Otherwise identical to train_n20.py (Phase 2: 4D, hidden_dim=32, warm restarts).
"""

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Override settings BEFORE importing train_n20's main
import train_n20

train_n20.GRAPH_SIZE         = 200
train_n20.CAPACITY           = 50
train_n20.BATCH_SIZE         = 512
train_n20.KNN_K              = 30
train_n20.ORTOOLS_EVAL_SIZE  = 500
train_n20.ORTOOLS_TIME_LIMIT = 5.0
train_n20.OUTPUT_DIR         = os.path.join(SCRIPT_DIR, "outputs", "n200")
train_n20.EPOCH_DIR          = os.path.join(train_n20.OUTPUT_DIR, "epochs")
train_n20.ARCHIVE_DIR        = os.path.join(train_n20.OUTPUT_DIR, "Archive")
train_n20.VAL_PATH           = os.path.join(SCRIPT_DIR, "datasets", "val_n200.pkl")
train_n20.BATCHES_PER_EPOCH  = train_n20.EPOCH_SIZE // train_n20.BATCH_SIZE
train_n20.TOTAL_OPT_STEPS    = train_n20.N_EPOCHS * train_n20.BATCHES_PER_EPOCH * 3 * 8

if __name__ == "__main__":
    train_n20.main()
