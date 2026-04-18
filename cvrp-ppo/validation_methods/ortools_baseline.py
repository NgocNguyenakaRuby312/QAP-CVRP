"""
validation_methods/ortools_baseline.py
========================================
Google OR-Tools CVRP baseline.

Install:
    pip install ortools

Usage:
    python ortools_baseline.py --n 10
    python ortools_baseline.py --n 20 --instances 200
"""

import os, sys, argparse, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_generator import load_dataset

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

DATASETS = {
    10:  ("datasets/val_n10.pkl",  20),
    20:  ("datasets/val_n20.pkl",  30),
    50:  ("datasets/val_n50.pkl",  40),
    100: ("datasets/val_n100.pkl", 50),
}

SCALE      = 100_000
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..")


def euclidean_int(a, b):
    return int(math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


def solve_instance(coords_np: np.ndarray, demands_np: np.ndarray,
                   capacity: int, time_limit: float) -> float:
    """
    Solve one CVRP instance with OR-Tools.

    Returns:
        tour_length: float (Euclidean, unscaled)
    """
    N = len(demands_np) - 1
    n_vehicles = N   # upper bound

    # Scale coordinates to integers
    coords_int = [(int(coords_np[i, 0] * SCALE),
                   int(coords_np[i, 1] * SCALE)) for i in range(N + 1)]

    # Distance matrix
    dist_matrix = [[euclidean_int(coords_int[i], coords_int[j])
                    for j in range(N + 1)] for i in range(N + 1)]

    manager = pywrapcp.RoutingIndexManager(N + 1, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def dist_cb(from_idx, to_idx):
        return dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Capacity constraint
    def demand_cb(from_idx):
        return int(demands_np[manager.IndexToNode(from_idx)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [capacity] * n_vehicles, True, "Capacity"
    )

    # Search parameters
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = int(time_limit)

    solution = routing.SolveWithParameters(params)

    if solution is None:
        return float("nan")

    # Extract total distance and convert back to float
    total = 0
    for v in range(n_vehicles):
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            next_idx = solution.Value(routing.NextVar(idx))
            total += routing.GetArcCostForVehicle(idx, next_idx, v)
            idx = next_idx

    return total / SCALE


def evaluate_ortools(n: int, n_instances: int = 100,
                     time_limit: float = 2.0) -> float:
    val_path, capacity = DATASETS[n]
    val_path = os.path.join(SCRIPT_DIR, val_path)

    coords_t, demands_t, _ = load_dataset(val_path, device="cpu")
    coords_np  = coords_t[:n_instances].numpy()
    demands_np = demands_t[:n_instances].numpy()

    total = 0.0
    valid = 0
    for i in range(len(coords_np)):
        d = solve_instance(coords_np[i], demands_np[i], capacity, time_limit)
        if not math.isnan(d):
            total += d
            valid += 1
        if (i + 1) % 10 == 0:
            avg = total / valid if valid else float("nan")
            print(f"  [{i+1}/{len(coords_np)}]  running avg: {avg:.4f}")

    return total / valid if valid else float("nan")


if __name__ == "__main__":
    if not ORTOOLS_AVAILABLE:
        print("OR-Tools not installed. Run:  pip install ortools")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,   choices=[10,20,50,100], default=10)
    parser.add_argument("--instances",  type=int,   default=100)
    parser.add_argument("--time_limit", type=float, default=2.0,
                        help="Seconds per instance")
    args = parser.parse_args()

    print(f"OR-Tools (GLS) — CVRP-{args.n}")
    print(f"  Instances  : {args.instances}")
    print(f"  Time/inst  : {args.time_limit}s")
    avg = evaluate_ortools(args.n, args.instances, args.time_limit)
    print(f"  Avg tour   : {avg:.4f}")
