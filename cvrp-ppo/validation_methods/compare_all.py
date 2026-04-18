"""
validation_methods/compare_all.py
===================================
Runs all baseline methods + QAP-DRL and prints a comparison table.

Usage:
    python compare_all.py --n 10
    python compare_all.py --n 10 --instances 200

Methods compared:
    1. Nearest Neighbour  (no extra install)
    2. PyVRP / HGS        (pip install pyvrp)
    3. OR-Tools / GLS     (pip install ortools)
    4. QAP-DRL (ours)     (loads best_model.pt)
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nearest_neighbor import evaluate_nn

try:
    from pyvrp_baseline import evaluate_pyvrp, PYVRP_AVAILABLE as PYVRP_OK
except ImportError:
    PYVRP_OK = False

try:
    from ortools_baseline import evaluate_ortools, ORTOOLS_AVAILABLE as ORTOOLS_OK
except ImportError:
    ORTOOLS_OK = False

import torch
from models.qap_policy import QAPPolicy
from training.evaluate import evaluate
from utils.data_generator import load_dataset

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..")

DATASETS = {
    10:  ("datasets/val_n10.pkl",  20, 4.84,  "outputs/n10/best_model.pt"),
    20:  ("datasets/val_n20.pkl",  30, 6.10,  "outputs/n20/best_model.pt"),
    50:  ("datasets/val_n50.pkl",  40, 10.38, "outputs/n50/best_model.pt"),
    100: ("datasets/val_n100.pkl", 50, 15.65, "outputs/n100/best_model.pt"),
}

LKH3_REF = {10: 4.84, 20: 6.10, 50: 10.38, 100: 15.65}


def evaluate_qap_drl(n: int, n_instances: int) -> float:
    val_path, capacity, _, model_path = DATASETS[n]
    model_path = os.path.join(SCRIPT_DIR, model_path)
    val_path   = os.path.join(SCRIPT_DIR, val_path)

    if not os.path.exists(model_path):
        return float("nan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = QAPPolicy()
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.to(device).eval()

    coords_t, demands_t, cap = load_dataset(val_path, device=str(device))
    val_instances = (
        coords_t[:n_instances],
        demands_t[:n_instances],
        cap,
    )
    return evaluate(policy, val_instances, device, greedy=True)


def run_comparison(n: int, instances: int, time_limit: float):
    lkh3 = LKH3_REF[n]

    print(f"\n{'='*62}")
    print(f"  Comparison — CVRP-{n}  ({instances} val instances)")
    print(f"{'='*62}")

    results = {}

    # 1. LKH3 (literature reference)
    results["LKH3 (literature)"] = lkh3
    print(f"  LKH3 (literature)  : {lkh3:.4f}  [reference]")

    # 2. Nearest Neighbour
    print("  Running Nearest Neighbour ...", end=" ", flush=True)
    nn = evaluate_nn(n, instances)
    results["Nearest Neighbour"] = nn
    print(f"{nn:.4f}")

    # 3. PyVRP
    if PYVRP_OK:
        print(f"  Running PyVRP (HGS, {time_limit}s/inst) ...")
        pv = evaluate_pyvrp(n, instances, time_limit)
        results["PyVRP (HGS)"] = pv
        print(f"  PyVRP (HGS)        : {pv:.4f}")
    else:
        print("  PyVRP              : not installed  (pip install pyvrp)")

    # 4. OR-Tools
    if ORTOOLS_OK:
        print(f"  Running OR-Tools (GLS, {time_limit}s/inst) ...")
        ot = evaluate_ortools(n, instances, time_limit)
        results["OR-Tools (GLS)"] = ot
        print(f"  OR-Tools (GLS)     : {ot:.4f}")
    else:
        print("  OR-Tools           : not installed  (pip install ortools)")

    # 5. QAP-DRL (ours)
    print("  Running QAP-DRL ...", end=" ", flush=True)
    drl = evaluate_qap_drl(n, instances)
    results["QAP-DRL (ours)"] = drl
    if not (drl != drl):  # nan check
        print(f"{drl:.4f}")
    else:
        print("no model found")

    # ── Final table ──────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  {'Method':<22} {'Avg Tour':>10} {'Gap vs LKH3':>12}")
    print(f"{'─'*62}")
    for name, val in results.items():
        if val != val:  # nan
            print(f"  {name:<22} {'N/A':>10} {'N/A':>12}")
        else:
            gap = 100.0 * (val - lkh3) / lkh3
            marker = " *" if name == "QAP-DRL (ours)" else ""
            print(f"  {name:<22} {val:>10.4f} {gap:>+11.2f}%{marker}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,   choices=[10,20,50,100], default=10)
    parser.add_argument("--instances",  type=int,   default=200,
                        help="Val instances per method")
    parser.add_argument("--time_limit", type=float, default=2.0,
                        help="Seconds per instance for PyVRP / OR-Tools")
    args = parser.parse_args()

    run_comparison(args.n, args.instances, args.time_limit)
