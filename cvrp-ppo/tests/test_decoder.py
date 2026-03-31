"""
tests/test_decoder.py — Decoder masking, feasibility, tour validity, and device checks.
Run: python tests/test_decoder.py   OR   pytest tests/test_decoder.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from encoder import FullEncoder
from decoder import QAPDecoder
from environment.cvrp_env import CVRPEnv

B, N, CAP = 4, 10, 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_episode():
    """Build model, run a full greedy episode, return (actions, state, demands, env)."""
    enc = FullEncoder(5, 2, 16, 5).to(device)
    dec = QAPDecoder(4, 2, 0.1).to(device)
    env = CVRPEnv(num_loc=N, device=str(device))

    coords = torch.rand(B, N + 1, 2, device=device)
    demands = torch.zeros(B, N + 1, device=device)
    demands[:, 1:] = torch.randint(1, 5, (B, N), device=device).float()

    state = env.reset({
        "coords":   coords,
        "demands":  demands,
        "capacity": torch.full((B,), float(CAP), device=device),
    })

    psi_prime, features, knn_indices = enc(state)
    actions_t, log_probs_t, tour_length = dec.rollout(
        psi_prime, state, knn_indices, env, greedy=True,
    )
    return actions_t, log_probs_t, tour_length, demands, coords


def test_depot_never_masked():
    enc = FullEncoder(5, 2, 16, 5).to(device)
    dec = QAPDecoder(4, 2, 0.1).to(device)
    coords = torch.rand(B, N + 1, 2, device=device)
    demands = torch.zeros(B, N + 1, device=device)
    demands[:, 1:] = torch.randint(1, 5, (B, N), device=device).float()
    state = {
        "coords": coords, "demands": demands,
        "capacity": torch.full((B,), float(CAP), device=device),
        "current_node": torch.zeros(B, dtype=torch.long, device=device),
        "used_capacity": torch.zeros(B, device=device),
        "visited": torch.zeros(B, N + 1, dtype=torch.bool, device=device),
        "action_mask": torch.ones(B, N + 1, dtype=torch.bool, device=device),
        "done": torch.zeros(B, dtype=torch.bool, device=device),
    }
    psi_prime, _, knn = enc(state)
    log_probs, mask = dec(state, psi_prime, knn, step=0, n_customers=N)
    assert mask[:, 0].sum() == 0, "depot must never be masked"


def test_infeasible_zero_prob():
    enc = FullEncoder(5, 2, 16, 5).to(device)
    dec = QAPDecoder(4, 2, 0.1).to(device)
    coords = torch.rand(B, N + 1, 2, device=device)
    demands = torch.zeros(B, N + 1, device=device)
    demands[:, 1:] = torch.randint(1, 5, (B, N), device=device).float()
    state = {
        "coords": coords, "demands": demands,
        "capacity": torch.full((B,), float(CAP), device=device),
        "current_node": torch.ones(B, dtype=torch.long, device=device),
        "used_capacity": torch.zeros(B, device=device),
        "visited": torch.zeros(B, N + 1, dtype=torch.bool, device=device),
        "action_mask": torch.ones(B, N + 1, dtype=torch.bool, device=device),
        "done": torch.zeros(B, dtype=torch.bool, device=device),
    }
    state["visited"][:, 1] = True
    state["visited"][:, 2] = True
    state["action_mask"] = ~state["visited"]
    state["action_mask"][:, 0] = True
    psi_prime, _, knn = enc(state)
    log_probs, mask = dec(state, psi_prime, knn, step=1, n_customers=N)
    probs = log_probs.exp()
    assert (probs[mask] < 1e-6).all(), "infeasible nodes must have ~0 probability"


def test_all_customers_visited():
    actions, _, _, demands, coords = _run_episode()
    for b in range(B):
        customers = [n.item() for n in actions[b] if n.item() != 0]
        assert len(customers) == N, f"batch {b}: visited {len(customers)}/{N}"
        assert len(set(customers)) == N, f"batch {b}: duplicate visits"


def test_no_capacity_violations():
    actions, _, _, demands, coords = _run_episode()
    for b in range(B):
        cap = float(CAP)
        for node in actions[b]:
            n = node.item()
            if n == 0:
                cap = float(CAP)
            else:
                cap -= demands[b, n].item()
                assert cap >= -1e-6, f"batch {b}: capacity violated at node {n}, cap={cap}"


def test_device():
    actions, _, tour_length, _, _ = _run_episode()
    assert actions.device.type == device.type
    assert tour_length.device.type == device.type


if __name__ == "__main__":
    print(f"Device: {device}")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    if device.type == "cuda":
        print(f"VRAM used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
