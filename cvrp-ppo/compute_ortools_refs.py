"""
compute_ortools_refs.py
========================
Compute OR-Tools (GLS) baseline tour lengths on the EXACT SAME validation
dataset and instance count that the training scripts use.

Consistency guarantee:
  - Loads val_nXX.pkl via load_dataset() — identical to training
  - Uses capacity FROM THE PKL FILE — same as training (not hardcoded)
  - Default --instances 1000 matches VAL_EVAL_SIZE in train_nXX.py
  - Saves val_path + capacity + n_instances to JSON for audit

Run ONCE per problem size before training:

    python compute_ortools_refs.py --n 20
    python compute_ortools_refs.py --n 50

Time estimates (CPU, GLS, 2s/instance):
    CVRP-20, 1000 inst : ~35 min
    CVRP-50, 1000 inst : ~35 min

To use fewer instances (faster, less precise):
    python compute_ortools_refs.py --n 20 --instances 200
    # ~7 min — still consistent because it uses instances [0:200] from the
    # same pkl that training evaluates as instances [0:1000].
    # Note this in your thesis: "OR-Tools ref computed on first 200 of 1000 val instances"

Results are saved to datasets/ortools_refs.json and loaded automatically
by train_n20.py and train_n50.py via utils/ortools_refs.py.
"""

import os, sys, argparse, json, math, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils.data_generator import load_dataset, get_capacity

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    ORTOOLS_OK = True
except ImportError:
    ORTOOLS_OK = False

# ── Single source of truth for val dataset paths ─────────────────────────────
# These MUST match the VAL_PATH constants in train_nXX.py.
VAL_PATHS = {
    10:  os.path.join(SCRIPT_DIR, "datasets", "val_n10.pkl"),
    20:  os.path.join(SCRIPT_DIR, "datasets", "val_n20.pkl"),
    50:  os.path.join(SCRIPT_DIR, "datasets", "val_n50.pkl"),
    100: os.path.join(SCRIPT_DIR, "datasets", "val_n100.pkl"),
}

# Must match VAL_EVAL_SIZE in train_nXX.py for full consistency.
# Can be reduced (e.g. 200) for a faster run — see module docstring.
DEFAULT_INSTANCES = 1000

REFS_PATH = os.path.join(SCRIPT_DIR, "datasets", "ortools_refs.json")
SCALE     = 100_000


# ── Solver ────────────────────────────────────────────────────────────────────

def euclidean_int(a, b):
    return int(math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


def solve_one(coords_np: np.ndarray, demands_np: np.ndarray,
              capacity: int, time_limit: float) -> float:
    """Solve one CVRP instance with OR-Tools GLS. Returns Euclidean tour length."""
    N = len(demands_np) - 1
    coords_int = [(int(coords_np[i, 0] * SCALE),
                   int(coords_np[i, 1] * SCALE)) for i in range(N + 1)]

    dist = [[euclidean_int(coords_int[i], coords_int[j])
             for j in range(N + 1)] for i in range(N + 1)]

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
    if sol is None:
        return float("nan")

    total = 0
    for v in range(N):
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            nxt = sol.Value(routing.NextVar(idx))
            total += routing.GetArcCostForVehicle(idx, nxt, v)
            idx = nxt
    return total / SCALE


# ── Main computation ──────────────────────────────────────────────────────────

def compute_ref(n: int, n_instances: int, time_limit: float) -> dict:
    """
    Solve n_instances from val_nXX.pkl and return stats dict.

    Capacity is read from the pkl file — same as training.
    """
    val_path = VAL_PATHS[n]

    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Validation dataset not found: {val_path}\n"
            f"Generate it first with: python utils/data_generator.py"
        )

    # Load exactly as training does — capacity from pkl, not hardcoded
    coords_t, demands_t, capacity = load_dataset(val_path, device="cpu")

    total_available = coords_t.shape[0]
    if n_instances > total_available:
        print(f"  WARNING: requested {n_instances} but only {total_available} available."
              f" Using {total_available}.")
        n_instances = total_available

    # Slice to first n_instances — same ordering as training's [:VAL_EVAL_SIZE]
    coords_np  = coords_t[:n_instances].numpy()    # [n_instances, N+1, 2]
    demands_np = demands_t[:n_instances].numpy()   # [n_instances, N+1]

    print(f"\nOR-Tools (GLS) — CVRP-{n}")
    print(f"  Val dataset : {val_path}")
    print(f"  Instances   : {n_instances}  (first {n_instances} of {total_available})")
    print(f"  Capacity    : {capacity}  (loaded from pkl — matches training)")
    print(f"  Time/inst   : {time_limit}s")
    print(f"  Est. total  : ~{n_instances * time_limit / 60:.0f} min")
    print()

    t0     = time.time()
    tours  = []
    failed = 0

    for i in range(n_instances):
        d = solve_one(coords_np[i], demands_np[i], capacity, time_limit)
        if math.isnan(d):
            failed += 1
        else:
            tours.append(d)

        if (i + 1) % 50 == 0 or (i + 1) == n_instances:
            avg     = sum(tours) / len(tours) if tours else float("nan")
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n_instances - i - 1)
            print(f"  [{i+1:>4}/{n_instances}]  avg={avg:.4f}  "
                  f"failed={failed}  elapsed={elapsed:.0f}s  ETA~{eta:.0f}s")

    mean_tour = sum(tours) / len(tours) if tours else float("nan")
    std_tour  = float(np.std(tours)) if tours else float("nan")

    print(f"\n  Final mean tour : {mean_tour:.4f}  (±{std_tour:.4f})")
    print(f"  Valid / total   : {len(tours)} / {n_instances}")

    return {
        "mean_tour":    round(mean_tour, 6),
        "std_tour":     round(std_tour,  6),
        "n_instances":  n_instances,
        "n_valid":      len(tours),
        "n_failed":     failed,
        "time_limit_s": time_limit,
        "capacity":     int(capacity),         # from pkl — same as training
        "val_path":     val_path,              # exact file used — audit trail
        "method":       "OR-Tools GLS (PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH)",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not ORTOOLS_OK:
        print("OR-Tools not installed.  Run:  pip install ortools")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Compute OR-Tools reference on the same val dataset as training."
    )
    parser.add_argument("--n", type=int, choices=[10, 20, 50, 100], required=True,
                        help="Graph size")
    parser.add_argument("--instances", type=int, default=DEFAULT_INSTANCES,
                        help=f"Number of instances (default {DEFAULT_INSTANCES} = VAL_EVAL_SIZE in training)")
    parser.add_argument("--time_limit", type=float, default=2.0,
                        help="OR-Tools GLS time budget per instance in seconds")
    args = parser.parse_args()

    # Consistency check: warn if different from training's VAL_EVAL_SIZE
    if args.instances != DEFAULT_INSTANCES:
        print(f"WARNING: --instances {args.instances} differs from training's "
              f"VAL_EVAL_SIZE={DEFAULT_INSTANCES}.")
        print(f"  The OR-Tools ref will be computed on a SUBSET of the training "
              f"validation set.")
        print(f"  This is acceptable but note it in your thesis.\n")

    stats = compute_ref(args.n, args.instances, args.time_limit)

    # Load existing refs, update entry for this n
    refs = {}
    if os.path.exists(REFS_PATH):
        try:
            with open(REFS_PATH) as f:
                refs = json.load(f)
        except Exception:
            refs = {}

    refs[str(args.n)] = stats

    with open(REFS_PATH, "w") as f:
        json.dump(refs, f, indent=2)

    print(f"\n  Saved to : {REFS_PATH}")
    print(f"  Entry    : CVRP-{args.n} → mean={stats['mean_tour']:.4f}")
    print(f"  Capacity : {stats['capacity']}  (from {os.path.basename(stats['val_path'])})")
    print()
    print("  Consistency check for thesis:")
    print(f"    Val file  : {stats['val_path']}")
    print(f"    Instances : {stats['n_instances']}")
    print(f"    Capacity  : {stats['capacity']}")
    print(f"    Training uses same file with first {DEFAULT_INSTANCES} instances.")
    if stats['n_instances'] == DEFAULT_INSTANCES:
        print(f"    FULL MATCH — OR-Tools and training use identical instance set.")
    else:
        print(f"    PARTIAL MATCH — OR-Tools used first {stats['n_instances']}, "
              f"training uses first {DEFAULT_INSTANCES}.")
    print()
    print("  Now run:  python train_n{}.py".format(args.n))


if __name__ == "__main__":
    main()
