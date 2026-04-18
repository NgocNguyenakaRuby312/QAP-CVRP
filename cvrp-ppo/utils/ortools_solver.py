"""
utils/ortools_solver.py
========================
Pure OR-Tools CVRP solver logic, shared by:
  - compute_ortools_refs.py  (standalone CLI)
  - utils/ortools_refs.py    (auto-compute inside train_nXX.py)

No side effects — just solve_one() and compute_and_save_ref().

v2 — stores per-instance tours + solve times to compute percentile
     distribution (p10/p25/p50/p75/p90) and timing stats.
"""

import math
import time
import json
import os
import numpy as np

SCALE     = 100_000
REFS_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "ortools_refs.json")

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    ORTOOLS_OK = True
except ImportError:
    ORTOOLS_OK = False


# ── Low-level solver ──────────────────────────────────────────────────────────

def _euclidean_int(a, b):
    return int(math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


def solve_one(coords_np: np.ndarray, demands_np: np.ndarray,
              capacity: int, time_limit: float) -> tuple:
    """
    Solve one CVRP instance with OR-Tools GLS.

    Args:
        coords_np:  [N+1, 2]  depot at index 0
        demands_np: [N+1]     depot demand = 0
        capacity:   int
        time_limit: float     seconds budget per instance

    Returns:
        (tour_length: float, solve_time: float)
        tour_length = nan if infeasible
    """
    t_start = time.time()

    N = len(demands_np) - 1
    coords_int = [
        (int(coords_np[i, 0] * SCALE), int(coords_np[i, 1] * SCALE))
        for i in range(N + 1)
    ]

    dist = [
        [_euclidean_int(coords_int[i], coords_int[j]) for j in range(N + 1)]
        for i in range(N + 1)
    ]

    manager = pywrapcp.RoutingIndexManager(N + 1, N, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(fi, ti):
        return dist[manager.IndexToNode(fi)][manager.IndexToNode(ti)]
    ti = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(ti)

    def dem_cb(fi):
        return int(demands_np[manager.IndexToNode(fi)])
    di = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(di, 0, [capacity] * N, True, "Cap")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = int(max(1, time_limit))

    sol = routing.SolveWithParameters(params)
    solve_time = time.time() - t_start

    if sol is None:
        return float("nan"), solve_time

    total = 0
    for v in range(N):
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            nxt = sol.Value(routing.NextVar(idx))
            total += routing.GetArcCostForVehicle(idx, nxt, v)
            idx = nxt

    return total / SCALE, solve_time


# ── Batch computation + save ──────────────────────────────────────────────────

def compute_and_save_ref(
    n: int,
    val_path: str,
    n_instances: int,
    capacity: int,
    coords_np: np.ndarray,
    demands_np: np.ndarray,
    time_limit: float = 2.0,
    silent: bool = False,
) -> dict:
    """
    Run OR-Tools on n_instances from the validation set, save result to
    datasets/ortools_refs.json, and return the stats dict.

    Stores: mean/std tour, percentiles (p10/p25/p50/p75/p90),
            timing stats (mean/max solve time per instance),
            n_valid/n_failed, capacity, method.

    Args:
        n:           graph size (key for JSON)
        val_path:    path to the pkl file (stored in JSON for audit)
        n_instances: number of instances solved
        capacity:    vehicle capacity (from pkl)
        coords_np:   [n_instances, N+1, 2]
        demands_np:  [n_instances, N+1]
        time_limit:  seconds per instance
        silent:      suppress progress output

    Returns:
        stats dict with all computed fields
    """
    if not ORTOOLS_OK:
        raise ImportError("OR-Tools not installed. Run:  pip install ortools")

    if not silent:
        est_min = n_instances * time_limit / 60
        print(f"\n{'='*68}")
        print(f"  Computing OR-Tools (GLS) reference — CVRP-{n}")
        print(f"{'='*68}")
        print(f"  Instances   : {n_instances}")
        print(f"  Capacity    : {capacity}  (from val pkl)")
        print(f"  Time/inst   : {time_limit}s")
        print(f"  Est. total  : ~{est_min:.0f} min")
        print(f"  (Training will start automatically when done)")
        print(f"{'='*68}\n")

    t0 = time.time()
    tours       = []
    solve_times = []
    failed      = 0

    for i in range(n_instances):
        d, st = solve_one(coords_np[i], demands_np[i], capacity, time_limit)
        solve_times.append(st)
        if math.isnan(d):
            failed += 1
        else:
            tours.append(d)

        if not silent and ((i + 1) % 25 == 0 or (i + 1) == n_instances):
            avg     = sum(tours) / len(tours) if tours else float("nan")
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n_instances - i - 1)
            print(
                f"  [{i+1:>4}/{n_instances}]  "
                f"avg = {avg:.4f}  "
                f"failed = {failed}  "
                f"elapsed = {elapsed:.0f}s  "
                f"ETA ~ {eta:.0f}s"
            )

    # ── Compute statistics ────────────────────────────────────────────────
    tours_arr = np.array(tours)
    mean_tour = float(tours_arr.mean())   if len(tours) > 0 else float("nan")
    std_tour  = float(tours_arr.std())    if len(tours) > 1 else 0.0
    p10       = float(np.percentile(tours_arr, 10))  if len(tours) > 0 else float("nan")
    p25       = float(np.percentile(tours_arr, 25))  if len(tours) > 0 else float("nan")
    p50       = float(np.percentile(tours_arr, 50))  if len(tours) > 0 else float("nan")
    p75       = float(np.percentile(tours_arr, 75))  if len(tours) > 0 else float("nan")
    p90       = float(np.percentile(tours_arr, 90))  if len(tours) > 0 else float("nan")
    t_mean    = float(np.mean(solve_times))
    t_max     = float(np.max(solve_times))
    n_time_limited = int(sum(1 for t in solve_times if t >= time_limit * 0.95))

    if not silent:
        cv = 100.0 * std_tour / mean_tour if mean_tour > 0 else 0.0
        print(f"\n  {'='*64}")
        print(f"  OR-Tools Reference Results — CVRP-{n}")
        print(f"  {'='*64}")
        print(f"  Mean tour length  : {mean_tour:.4f}")
        print(f"  Std deviation     : {std_tour:.4f}  (CV = {cv:.1f}%)")
        print(f"  Expected range    : {mean_tour - 2*std_tour:.2f} – {mean_tour + 2*std_tour:.2f}  (mean ± 2σ)")
        print(f"  {'─'*64}")
        print(f"  Percentiles:")
        print(f"    p10 = {p10:.4f}  p25 = {p25:.4f}  p50 = {p50:.4f}  p75 = {p75:.4f}  p90 = {p90:.4f}")
        print(f"  {'─'*64}")
        print(f"  Valid / total     : {len(tours)} / {n_instances}  ({failed} failed)")
        print(f"  Solve time/inst   : mean = {t_mean:.2f}s  max = {t_max:.2f}s")
        print(f"  Time-limited inst : {n_time_limited} / {n_instances}  (≥95% of budget used)")
        print(f"  {'─'*64}")
        print(f"  5% gap target     : ≤ {mean_tour * 1.05:.4f}  (model must beat this)")
        print(f"  {'='*64}\n")

    stats = {
        "mean_tour":       round(mean_tour, 6),
        "std_tour":        round(std_tour,  6),
        "p10":             round(p10,  4),
        "p25":             round(p25,  4),
        "p50":             round(p50,  4),
        "p75":             round(p75,  4),
        "p90":             round(p90,  4),
        "n_instances":     n_instances,
        "n_valid":         len(tours),
        "n_failed":        failed,
        "n_time_limited":  n_time_limited,
        "mean_solve_time": round(t_mean, 3),
        "max_solve_time":  round(t_max,  3),
        "time_limit_s":    time_limit,
        "capacity":        int(capacity),
        "val_path":        val_path,
        "method":          "OR-Tools GLS (PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH)",
    }

    # Persist to JSON
    refs = {}
    refs_path = os.path.normpath(REFS_PATH)
    if os.path.exists(refs_path):
        try:
            with open(refs_path) as f:
                refs = json.load(f)
        except Exception:
            refs = {}

    refs[str(n)] = stats
    os.makedirs(os.path.dirname(refs_path), exist_ok=True)
    with open(refs_path, "w") as f:
        json.dump(refs, f, indent=2)

    if not silent:
        print(f"  Saved → {refs_path}")

    return stats
