"""
environment/cvrp_env.py
=======================
Constructive CVRP environment with dict-based state interface.

State dict keys:
    coords, demands, capacity, current_node, used_capacity,
    visited, action_mask, step_count, done
"""

import torch


class CVRPEnv:
    """
    Vectorised CVRP environment operating on batched tensors.

    Args:
        num_loc: N customers per instance (excluding depot)
        device:  torch device string
    """

    def __init__(self, num_loc: int = 100, device: str = "cpu"):
        self.num_loc = num_loc
        self.device  = device

    def reset(self, instance: dict) -> dict:
        """
        Initialise episode from a batch of CVRP instances.

        Args:
            instance: dict with coords [B,N+1,2], demands [B,N+1], capacity [B] or scalar

        Returns:
            state dict
        """
        coords  = instance["coords"]
        demands = instance["demands"]
        cap     = instance["capacity"]
        B       = coords.shape[0]
        device  = coords.device

        if isinstance(cap, (int, float)):
            cap = torch.full((B,), float(cap), device=device)
        elif cap.dim() == 0:
            cap = cap.expand(B)

        state = {
            "coords":        coords,                                   # [B, N+1, 2]
            "demands":       demands,                                  # [B, N+1]
            "capacity":      cap,                                      # [B]
            "current_node":  torch.zeros(B, dtype=torch.long, device=device),
            "used_capacity": torch.zeros(B, device=device),
            "visited":       torch.zeros(B, self.num_loc + 1, dtype=torch.bool, device=device),
            "step_count":    torch.zeros(B, dtype=torch.long, device=device),
            "done":          torch.zeros(B, dtype=torch.bool, device=device),
        }
        state["action_mask"] = self._compute_mask(state)
        return state

    def step(self, state: dict, actions: torch.Tensor) -> dict:
        """
        Apply actions and return next state dict.

        Args:
            state:   current state dict
            actions: [B] selected node indices

        Returns:
            next state dict
        """
        demands       = state["demands"]
        capacity      = state["capacity"]
        used_capacity = state["used_capacity"]
        visited       = state["visited"].clone()

        selected_demand = demands.gather(1, actions.unsqueeze(1)).squeeze(1)

        at_depot = (actions == 0)
        new_used = torch.where(at_depot, torch.zeros_like(used_capacity),
                               used_capacity + selected_demand)        # P6: reset at depot

        visited.scatter_(1, actions.unsqueeze(1), True)
        visited[:, 0] = False                                          # depot never permanently visited

        done = visited[:, 1:].all(dim=1)

        next_state = {
            **state,
            "current_node":  actions,
            "used_capacity": new_used,
            "visited":       visited,
            "step_count":    state["step_count"] + 1,
            "done":          done,
        }
        next_state["action_mask"] = self._compute_mask(next_state)
        return next_state

    def _compute_mask(self, state: dict) -> torch.Tensor:
        """True = node can be selected. Depot always reachable (P3)."""
        remaining = (state["capacity"] - state["used_capacity"]).unsqueeze(1)  # [B, 1]
        exceeds   = state["demands"] > remaining                       # [B, N+1]
        mask      = ~state["visited"] & ~exceeds                       # [B, N+1]
        mask[:, 0] = True                                              # P3: depot always reachable

        # Prevent empty trips: if at depot and feasible customers exist, block depot
        at_depot = (state["current_node"] == 0)                        # [B]
        has_customer = mask[:, 1:].any(dim=1)                          # [B]
        block_depot = at_depot & has_customer                          # [B]
        mask[:, 0] = mask[:, 0] & ~block_depot                        # block depot for those
        return mask

    @staticmethod
    def get_reward(state: dict, actions: torch.Tensor) -> torch.Tensor:
        """R = -total_distance. state must have 'coords' key."""
        coords = state["coords"] if isinstance(state, dict) else state.coords
        B, T = actions.shape
        idx   = actions.unsqueeze(-1).expand(B, T, 2)
        route = coords.gather(1, idx)
        depot = coords[:, 0:1, :]
        full  = torch.cat([depot, route, depot], dim=1)
        return -(full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)
