"""
validation_methods/pyvrp_baseline.py
======================================
PyVRP (Hybrid Genetic Search) baseline for CVRP.

Install:
    pip install pyvrp

Usage:
    python pyvrp_baseline.py --n 10
    python pyvrp_baseline.py --n 20 --instances 200 --time_limit 5
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_generator import load_dataset

try:
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    PYVRP_AVAILABLE = True
except ImportError:
    PYVRP_AVAILABLE = False

DATASETS = {
    10:  ("datasets/val_n10.pkl",  20),
    20:  ("datasets/val_n20.pkl",  30),
    50:  ("datasets/val_n50.pkl",  40),
    100: ("datasets/val_n100.pkl", 50),
}

SCALE = 100_000   # float coords → int (PyVRP requires integers)
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..")


def solve_instance(coords_np: np.ndarray, demands_np: np.ndarray,
                   capacity: int, time_limit: float) -> float:
    """
    Solve one CVRP instance with PyVRP.

    Args:
        coords_np:  [N+1, 2]  float coordinates
        demands_np: [N+1]     integer demands
        capacity:   int
        time_limit: float     seconds per instance

    Returns:
        tour_length: float (Euclidean, unscaled)
    """
    N = len(demands_np) - 1

    m = Model()

    # Depot (index 0)
    m.add_depot(
        x=int(coords_np[0, 0] * SCALE),
        y=int(coords_np[0, 1] * SCALE),
    )

    # Customers (indices 1..N)
    for i in range(1, N + 1):
        m.add_client(
            x=int(coords_np[i, 0] * SCALE),
            y=int(coords_np[i, 1] * SCALE),
            delivery=int(demands_np[i]),
        )

    # Vehicle type: enough vehicles to always find a solution
    m.add_vehicle_type(num_available=N, capacity=capacity)

    result = m.solve(stop=MaxRuntime(time_limit), display=False)

    # Distance is in scaled integer units → convert back to float
    return result.best.distance() / SCALE


def evaluate_pyvrp(n: int, n_instances: int = 100,
                   time_limit: float = 2.0) -> float:
    val_path, capacity = DATASETS[n]
    val_path = os.path.join(SCRIPT_DIR, val_path)

    coords_t, demands_t, _ = load_dataset(val_path, device="cpu")
    coords_np  = coords_t[:n_instances].numpy()
    demands_np = demands_t[:n_instances].numpy()

    total = 0.0
    for i in range(len(coords_np)):
        total += solve_instance(coords_np[i], demands_np[i], capacity, time_limit)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(coords_np)}]  running avg: {total/(i+1):.4f}")

    return total / len(coords_np)


if __name__ == "__main__":
    if not PYVRP_AVAILABLE:
        print("PyVRP not installed. Run:  pip install pyvrp")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,   choices=[10,20,50,100], default=10)
    parser.add_argument("--instances",  type=int,   default=100,
                        help="Number of val instances to solve (fewer = faster)")
    parser.add_argument("--time_limit", type=float, default=2.0,
                        help="Seconds per instance")
    args = parser.parse_args()

    print(f"PyVRP (HGS) — CVRP-{args.n}")
    print(f"  Instances  : {args.instances}")
    print(f"  Time/inst  : {args.time_limit}s")
    avg = evaluate_pyvrp(args.n, args.instances, args.time_limit)
    print(f"  Avg tour   : {avg:.4f}")
