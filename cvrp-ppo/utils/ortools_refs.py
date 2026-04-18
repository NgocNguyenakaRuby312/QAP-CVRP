"""
utils/ortools_refs.py
======================
Load OR-Tools reference tour lengths, auto-computing them if needed.

Primary API used by train_nXX.py:

    ORTOOLS_REF, ORTOOLS_SOURCE = ensure_ortools_ref(
        n          = GRAPH_SIZE,
        val_path   = VAL_PATH,
        n_instances= VAL_EVAL_SIZE,
        coords_t   = coords_t,
        demands_t  = demands_t,
        capacity   = val_cap,
        time_limit = ORTOOLS_TIME_LIMIT,
        output_dir = OUTPUT_DIR,   # optional — for showing current best model gap
    )

Behaviour:
  - Cached and valid → print rich banner, return instantly (< 1 ms).
  - Missing/stale   → compute via OR-Tools GLS (prints progress), save, return.
  - OR-Tools absent → return fallback estimate with warning.

v2 — rich banner prints mean, std, CV%, ±2σ range, percentile distribution,
     timing stats, 5% gap target, and current best model gap if available.
"""

import os
import json

_REFS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "datasets", "ortools_refs.json")
)

_FALLBACK = {
    "10":  5.02,
    "20":  6.58,
    "50":  11.24,
    "100": 16.90,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_json() -> dict:
    if os.path.exists(_REFS_PATH):
        try:
            with open(_REFS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _entry_is_valid(entry: dict, val_path: str, n_instances: int) -> bool:
    stored_path = os.path.normpath(entry.get("val_path", ""))
    stored_n    = int(entry.get("n_instances", 0))
    return (
        stored_path == os.path.normpath(val_path)
        and stored_n == n_instances
    )


def _print_banner(r: dict, n: int, source_tag: str, output_dir: str = None):
    """
    Print the full OR-Tools reference banner before training starts.

    Prints all cached statistics:
      - mean, std, CV%, ±2σ range
      - percentile distribution (p10/p25/p50/p75/p90) if available
      - n_valid/n_failed, timing stats if available
      - 5% gap target (= mean × 1.05)
      - current best model gap vs OR-Tools (if best_model.pt exists in output_dir)
    """
    mean   = float(r["mean_tour"])
    std    = float(r.get("std_tour", 0.0))
    cv     = 100.0 * std / mean if mean > 0 else 0.0
    n_inst = int(r.get("n_instances", 0))
    n_val  = int(r.get("n_valid", n_inst))
    n_fail = int(r.get("n_failed", 0))
    tlim   = float(r.get("time_limit_s", 0))
    target = mean * 1.05

    # Optional fields added by ortools_solver v2
    p10   = r.get("p10")
    p25   = r.get("p25")
    p50   = r.get("p50")
    p75   = r.get("p75")
    p90   = r.get("p90")
    t_mean = r.get("mean_solve_time")
    t_max  = r.get("max_solve_time")
    n_tl   = r.get("n_time_limited")

    W = 68
    print(f"\n  {'='*W}")
    print(f"  OR-Tools Reference — CVRP-{n}  [{source_tag}]")
    print(f"  {'='*W}")
    print(f"  Mean tour length  : {mean:.4f}")
    print(f"  Std deviation     : {std:.4f}  (CV = {cv:.1f}%)")
    print(f"  Expected range    : {mean - 2*std:.2f} – {mean + 2*std:.2f}  (mean ± 2σ)")

    if all(x is not None for x in [p10, p25, p50, p75, p90]):
        print(f"  {'─'*W}")
        print(f"  Percentile distribution:")
        print(f"    p10 = {p10:.4f}  p25 = {p25:.4f}  p50 = {p50:.4f}"
              f"  p75 = {p75:.4f}  p90 = {p90:.4f}")

    print(f"  {'─'*W}")
    print(f"  Valid / total     : {n_val} / {n_inst}  ({n_fail} failed)")

    if t_mean is not None and t_max is not None:
        tl_str = f"  ({n_tl}/{n_inst} hit time limit)" if n_tl is not None else ""
        print(f"  Solve time/inst   : mean = {t_mean:.2f}s  max = {t_max:.2f}s{tl_str}")

    print(f"  {'─'*W}")
    print(f"  5% gap target     : ≤ {target:.4f}  (tour length the model must beat)")

    # Show current best model gap if checkpoint exists
    if output_dir is not None:
        best_log = os.path.join(output_dir, "train_log.jsonl")
        if os.path.exists(best_log):
            try:
                best_tour = None
                with open(best_log, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        bt = row.get("best_tour")
                        if bt is not None:
                            best_tour = float(bt)
                if best_tour is not None:
                    gap = 100.0 * (best_tour - mean) / mean
                    status = "✓ BELOW TARGET" if best_tour <= target else "above target"
                    print(f"  Current best      : {best_tour:.4f}  "
                          f"(gap = {gap:+.1f}%)  [{status}]")
            except Exception:
                pass

    print(f"  {'='*W}\n")


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_ortools_ref(
    n:           int,
    val_path:    str,
    n_instances: int,
    coords_t,
    demands_t,
    capacity:    int,
    time_limit:  float = 2.0,
    output_dir:  str   = None,   # optional: for showing current best model gap
) -> tuple:
    """
    Return (mean_tour: float, source_str: str) for problem size n.

    Fast path  — entry already exists and matches current config → instant return.
    Compute path — entry missing or stale → run OR-Tools, save, return.
    Fallback   — OR-Tools not installed → return estimate with warning.
    """
    key  = str(n)
    refs = _load_json()

    # ── Fast path: valid cached entry ────────────────────────────────────────
    if key in refs and _entry_is_valid(refs[key], val_path, n_instances):
        r   = refs[key]
        src = (
            f"OR-Tools GLS  ({r['n_instances']} inst, "
            f"{r['time_limit_s']}s each)  [cached]"
        )
        _print_banner(r, n, "cached", output_dir=output_dir)
        return float(r["mean_tour"]), src

    # ── Check OR-Tools is available ──────────────────────────────────────────
    try:
        from utils.ortools_solver import ORTOOLS_OK, compute_and_save_ref
    except ImportError:
        ORTOOLS_OK = False

    if not ORTOOLS_OK:
        fallback = _FALLBACK.get(key, float("nan"))
        print(
            f"\n[ortools_refs] WARNING: OR-Tools not installed.\n"
            f"  Using fallback estimate for CVRP-{n}: {fallback:.2f}\n"
            f"  Install with:  pip install ortools\n"
        )
        return fallback, f"OR-Tools est. (fallback — install ortools)"

    # ── Compute path: entry missing or stale ─────────────────────────────────
    if key in refs and not _entry_is_valid(refs[key], val_path, n_instances):
        stored_n = refs[key].get("n_instances", "?")
        print(
            f"\n[ortools_refs] Stale reference for CVRP-{n}:\n"
            f"  Stored n_instances = {stored_n}, "
            f"current VAL_EVAL_SIZE = {n_instances}.\n"
            f"  Recomputing...\n"
        )

    import numpy as np
    coords_np  = coords_t[:n_instances].cpu().numpy()
    demands_np = demands_t[:n_instances].cpu().numpy()

    stats = compute_and_save_ref(
        n           = n,
        val_path    = val_path,
        n_instances = n_instances,
        capacity    = int(capacity),
        coords_np   = coords_np,
        demands_np  = demands_np,
        time_limit  = time_limit,
        silent      = False,
    )

    # Print banner after fresh computation
    _print_banner(stats, n, "just computed", output_dir=output_dir)

    mean_tour = float(stats["mean_tour"])
    src = (
        f"OR-Tools GLS  ({stats['n_instances']} inst, "
        f"{stats['time_limit_s']}s each)  [just computed]"
    )
    return mean_tour, src


# ── Legacy helpers ────────────────────────────────────────────────────────────

def load_ortools_ref(n: int) -> tuple:
    key  = str(n)
    refs = _load_json()
    if key in refs:
        r = refs[key]
        return float(r["mean_tour"]), int(r.get("n_instances", 0)), float(r.get("time_limit_s", 0.0))
    fallback = _FALLBACK.get(key, float("nan"))
    if fallback == fallback:
        print(
            f"[ortools_refs] WARNING: No ref for CVRP-{n}. "
            f"Using fallback estimate {fallback:.2f}. "
            f"Run train_n{n}.py to auto-compute."
        )
    return fallback, 0, 0.0


def has_computed_ref(n: int) -> bool:
    return str(n) in _load_json()


def verify_consistency(n: int, expected_val_path: str,
                       expected_n_instances: int) -> bool:
    refs = _load_json()
    key  = str(n)
    if key not in refs:
        return True
    r  = refs[key]
    ok = True
    stored_path = os.path.normpath(r.get("val_path", ""))
    if stored_path and stored_path != os.path.normpath(expected_val_path):
        print(
            f"[ortools_refs] PATH MISMATCH for CVRP-{n}:\n"
            f"  Stored  : {r.get('val_path')}\n"
            f"  Current : {expected_val_path}"
        )
        ok = False
    stored_n = int(r.get("n_instances", 0))
    if stored_n and stored_n != expected_n_instances:
        print(
            f"[ortools_refs] INSTANCE COUNT NOTE for CVRP-{n}: "
            f"ref used {stored_n} inst, training uses {expected_n_instances}."
        )
    return ok
