"""
environment/state.py
====================
State representation for the constructive CVRP environment.

The environment uses a plain dict (not a class) for state. All keys:

    coords        [B, N+1, 2]   float32  depot at index 0
    demands       [B, N+1]      float32  depot demand = 0
    capacity      [B]           float32  vehicle capacity C
    current_node  [B]           int64    current vehicle position
    used_capacity [B]           float32  demand used on current route
    visited       [B, N+1]      bool     True = customer already visited
                                         depot (index 0) is NEVER True
    action_mask   [B, N+1]      bool     True = FEASIBLE (can select)
                                         NOTE: this is True=feasible,
                                         opposite of masked_fill convention.
                                         Decoder inverts: mask = ~action_mask
    step_count    [B]           int64    steps taken so far
    done          [B]           bool     True = all customers visited

Episode ends when done.all() is True.
New route starts when agent selects depot (index 0).
Depot action resets used_capacity to 0.

StateCVRP is also provided as a class-based representation (used by tests).
"""

import torch


class StateCVRP:
    """
    Class-based CVRP state for use in tests and alternative interfaces.

    Attributes:
        coords        [B, N+1, 2]  float32
        demands       [B, N+1]     float32   depot demand = 0
        capacity      float        vehicle capacity C
        cur_node      [B]          int64     current vehicle position
        used_capacity [B]          float32   demand used on current route
        visited       [B, N+1]    bool      True = customer visited; depot always False
        lengths       [B]          float32   cumulative tour length so far
        step          int          decode steps taken
    """

    def __init__(
        self,
        coords:        torch.Tensor,
        demands:       torch.Tensor,
        capacity:      float,
        cur_node:      torch.Tensor,
        used_capacity: torch.Tensor,
        visited:       torch.Tensor,
        lengths:       torch.Tensor,
        step:          int,
    ):
        self.coords        = coords
        self.demands       = demands
        self.capacity      = capacity
        self.cur_node      = cur_node
        self.used_capacity = used_capacity
        self.visited       = visited
        self.lengths       = lengths
        self.step          = step

    @classmethod
    def initialize(cls, coords: torch.Tensor, demands: torch.Tensor, capacity) -> "StateCVRP":
        """
        Create initial state: all vehicles at depot, zero capacity used.

        Args:
            coords:   [B, N+1, 2]
            demands:  [B, N+1]
            capacity: scalar float or int

        Returns:
            StateCVRP at step 0
        """
        B      = coords.size(0)
        N1     = coords.size(1)
        device = coords.device
        return cls(
            coords        = coords,
            demands       = demands,
            capacity      = float(capacity),
            cur_node      = torch.zeros(B, dtype=torch.long, device=device),
            used_capacity = torch.zeros(B, device=device),
            visited       = torch.zeros(B, N1, dtype=torch.bool, device=device),
            lengths       = torch.zeros(B, device=device),
            step          = 0,
        )

    def update(self, action: torch.Tensor):
        """
        Transition to next state by selecting nodes in `action`.

        Args:
            action: [B]  node indices (0 = depot)

        Returns:
            (new_state, step_cost)   step_cost [B] euclidean distance travelled
        """
        B      = action.shape[0]
        device = action.device

        # Step cost: distance from current node to selected node
        arange      = torch.arange(B, device=device)
        cur_coords  = self.coords[arange, self.cur_node]        # [B, 2]
        next_coords = self.coords[arange, action]               # [B, 2]
        step_cost   = (next_coords - cur_coords).norm(dim=-1)   # [B]

        # Update visited: scatter True at action positions
        visited_new = self.visited.clone()
        visited_new.scatter_(1, action.unsqueeze(1),
                             torch.ones(B, 1, dtype=torch.bool, device=device))
        visited_new[:, 0] = False                               # depot never permanently visited

        # Update used capacity (reset to 0 when returning to depot)
        at_depot     = (action == 0)                            # [B]
        demand_taken = self.demands[arange, action]             # [B]
        new_used     = torch.where(
            at_depot,
            torch.zeros_like(self.used_capacity),
            self.used_capacity + demand_taken,
        )

        new_state = StateCVRP(
            coords        = self.coords,
            demands       = self.demands,
            capacity      = self.capacity,
            cur_node      = action.clone(),
            used_capacity = new_used,
            visited       = visited_new,
            lengths       = self.lengths + step_cost,
            step          = self.step + 1,
        )
        return new_state, step_cost

    def get_feasible_mask(self) -> torch.Tensor:
        """
        Returns infeasibility mask: True = node CANNOT be selected.

        A customer is infeasible if already visited or its demand exceeds
        remaining capacity. Depot (index 0) is always feasible.

        Returns:
            mask [B, N+1]  bool  True = infeasible
        """
        remaining = self.capacity - self.used_capacity          # [B]
        exceeds   = self.demands > remaining.unsqueeze(1)       # [B, N+1]
        mask      = self.visited | exceeds                      # [B, N+1]
        mask[:, 0] = False                                      # depot always feasible
        return mask

    def all_finished(self) -> bool:
        """True when all customers in every instance have been visited."""
        return bool(self.visited[:, 1:].all())
