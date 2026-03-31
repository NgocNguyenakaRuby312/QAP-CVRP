"""
Tests for Phase 1: seed, data_generator, state, cvrp_env.
Run: python -m pytest cvrp-ppo/tests/test_env.py -v
"""

import sys
import os

# Add cvrp-ppo directory to path so we can import utils/environment as top-level packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from utils.seed import set_seed
from utils.data_generator import generate_instances, get_capacity, CAPACITY_MAP
from environment.state import StateCVRP
from environment.cvrp_env import CVRPEnv


# ── seed ──────────────────────────────────────────────────────────────────────

def test_seed_reproducibility():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b), "set_seed should produce identical sequences"


# ── data_generator ────────────────────────────────────────────────────────────

def test_generate_shapes():
    for N in [20, 50, 100]:
        coords, demands, cap = generate_instances(8, N)
        assert coords.shape == (8, N + 1, 2), f"coords shape wrong for N={N}"
        assert demands.shape == (8, N + 1), f"demands shape wrong for N={N}"
        assert cap == CAPACITY_MAP[N], f"capacity wrong for N={N}"


def test_depot_demand_zero():
    coords, demands, cap = generate_instances(16, 20)
    assert (demands[:, 0] == 0).all(), "Depot demand must be 0"


def test_customer_demands_range():
    coords, demands, cap = generate_instances(64, 50)
    cust = demands[:, 1:]  # customer demands
    assert (cust >= 1).all() and (cust <= 9).all(), "Customer demands must be in [1,9]"


def test_coords_in_unit_square():
    coords, demands, cap = generate_instances(32, 20)
    assert (coords >= 0).all() and (coords <= 1).all(), "Coords must be in [0,1]"


def test_capacity_map():
    assert get_capacity(20) == 30
    assert get_capacity(50) == 40
    assert get_capacity(100) == 50


# ── StateCVRP ─────────────────────────────────────────────────────────────────

def _make_instance(batch=4, N=5, cap=20):
    """Small deterministic instance for testing."""
    set_seed(0)
    coords = torch.rand(batch, N + 1, 2)
    demands = torch.zeros(batch, N + 1)
    demands[:, 1:] = torch.randint(1, 10, (batch, N)).float()
    return coords, demands, cap


def test_state_initialize():
    coords, demands, cap = _make_instance()
    state = StateCVRP.initialize(coords, demands, cap)

    assert state.cur_node.shape == (4,)
    assert (state.cur_node == 0).all(), "All vehicles start at depot"
    assert (state.used_capacity == 0).all()
    assert not state.visited.any(), "No node should be visited initially"
    assert (state.lengths == 0).all()
    assert state.step == 0


def test_state_update_customer():
    coords, demands, cap = _make_instance()
    state = StateCVRP.initialize(coords, demands, cap)

    # Visit customer 1
    action = torch.ones(4, dtype=torch.long)
    new_state, step_cost = state.update(action)

    assert (new_state.cur_node == 1).all()
    assert new_state.visited[:, 1].all(), "Customer 1 should be visited"
    assert not new_state.visited[:, 0].any(), "Depot should not be marked visited"
    assert (step_cost > 0).all(), "Step cost should be positive"
    assert (new_state.used_capacity == demands[:, 1]).all()
    assert new_state.step == 1


def test_state_update_depot_resets_capacity():
    coords, demands, cap = _make_instance()
    state = StateCVRP.initialize(coords, demands, cap)

    # Visit customer 1 then return to depot
    action1 = torch.ones(4, dtype=torch.long)
    state, _ = state.update(action1)
    assert (state.used_capacity > 0).all()

    action0 = torch.zeros(4, dtype=torch.long)
    state, _ = state.update(action0)
    assert (state.used_capacity == 0).all(), "Returning to depot resets capacity"


def test_feasible_mask_visited():
    coords, demands, cap = _make_instance()
    state = StateCVRP.initialize(coords, demands, cap)

    # Visit customer 1
    action = torch.ones(4, dtype=torch.long)
    state, _ = state.update(action)

    mask = state.get_feasible_mask()
    assert mask[:, 1].all(), "Visited customer should be infeasible"
    assert not mask[:, 0].any(), "Depot always feasible"


def test_feasible_mask_capacity():
    """Customer whose demand exceeds remaining capacity should be masked."""
    batch = 2
    N = 3
    coords = torch.rand(batch, N + 1, 2)
    demands = torch.zeros(batch, N + 1)
    demands[:, 1] = 5.0
    demands[:, 2] = 8.0
    demands[:, 3] = 3.0
    cap = 10.0

    state = StateCVRP.initialize(coords, demands, cap)
    # Visit customer 1 (uses 5 capacity, remaining = 5)
    action = torch.ones(batch, dtype=torch.long)
    state, _ = state.update(action)

    mask = state.get_feasible_mask()
    # Customer 2 needs 8 > 5 remaining → infeasible
    assert mask[:, 2].all(), "Customer with demand > remaining cap should be infeasible"
    # Customer 3 needs 3 ≤ 5 remaining → feasible
    assert not mask[:, 3].any(), "Customer with demand ≤ remaining cap should be feasible"


def test_all_finished():
    batch, N = 2, 3
    coords = torch.rand(batch, N + 1, 2)
    demands = torch.zeros(batch, N + 1)
    demands[:, 1:] = 1.0
    cap = 100.0

    state = StateCVRP.initialize(coords, demands, cap)
    assert not state.all_finished()

    # Visit all customers
    for i in range(1, N + 1):
        action = torch.full((batch,), i, dtype=torch.long)
        state, _ = state.update(action)

    assert state.all_finished(), "Should be finished after visiting all customers"


# ── CVRPEnv ───────────────────────────────────────────────────────────────────

def test_env_reset_and_step():
    N = 5
    batch = 4
    cap = 20
    set_seed(0)
    coords = torch.rand(batch, N + 1, 2)
    demands = torch.zeros(batch, N + 1)
    demands[:, 1:] = torch.randint(1, 5, (batch, N)).float()

    env = CVRPEnv(num_loc=N)
    instance = {"coords": coords, "demands": demands, "capacity": cap}
    state = env.reset(instance)

    assert isinstance(state, dict)
    assert (state["current_node"] == 0).all()

    # Take a step
    action = torch.ones(batch, dtype=torch.long)
    state = env.step(state, action)
    assert state["step_count"].min().item() == 1
    assert not state["done"].all()


def test_env_full_episode():
    """Run a full greedy episode selecting nearest feasible customer."""
    N = 10
    batch = 2
    cap = 20
    set_seed(123)
    coords = torch.rand(batch, N + 1, 2)
    demands = torch.zeros(batch, N + 1)
    demands[:, 1:] = 2.0  # uniform demand = 2, cap = 20 -> 10 per route

    env = CVRPEnv(num_loc=N)
    instance = {"coords": coords, "demands": demands, "capacity": cap}
    state = env.reset(instance)

    actions_taken = []
    max_steps = 3 * N

    for _ in range(max_steps):
        mask = ~state["action_mask"]  # True = infeasible
        feasible = state["action_mask"]  # True = feasible

        cur_coords = coords[torch.arange(batch), state["current_node"]]
        dists = (coords - cur_coords.unsqueeze(1)).norm(dim=-1)
        dists[mask] = float("inf")

        has_customer = feasible.clone()
        has_customer[:, 0] = False
        for b in range(batch):
            if has_customer[b].any():
                dists[b, 0] = float("inf")

        action = dists.argmin(dim=-1)
        actions_taken.append(action)

        state = env.step(state, action)

        if state["done"].all():
            break

    assert state["done"].all(), "Episode should complete within max_steps"
    assert state["visited"][:, 1:].all(), "All customers should be visited"


def test_env_get_reward_static():
    """Test static get_reward method for tour length computation."""
    batch = 2
    # Simple geometry: depot at (0,0), customer 1 at (1,0), customer 2 at (1,1)
    coords = torch.zeros(batch, 3, 2)
    coords[:, 1, 0] = 1.0  # customer 1 at (1, 0)
    coords[:, 2, 0] = 1.0  # customer 2 at (1, 1)
    coords[:, 2, 1] = 1.0

    # Tour: depot → 1 → 2
    actions = torch.tensor([[1, 2], [1, 2]])
    reward = CVRPEnv.get_reward({"coords": coords}, actions)

    # Expected: depot(0,0)→1(1,0)=1 + 1(1,0)→2(1,1)=1 + 2(1,1)→depot(0,0)=sqrt(2)
    expected = -(1.0 + 1.0 + 2**0.5)
    assert torch.allclose(reward, torch.full((batch,), expected), atol=1e-5)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        sys.exit(1)
