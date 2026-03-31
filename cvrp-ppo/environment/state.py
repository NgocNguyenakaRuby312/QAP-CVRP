import torch


class StateCVRP:
    """
    Constructive MDP state for CVRP.

    Static fields (set at episode start):
        coords:      [batch, N+1, 2]  — depot at index 0
        demands:     [batch, N+1]     — depot demand = 0
        capacity:    float            — vehicle capacity Q
        num_nodes:   int              — N+1 (including depot)

    Dynamic fields (updated each step):
        cur_node:      [batch]        — current vehicle position (node index)
        used_capacity: [batch]        — demand consumed on current route
        visited:       [batch, N+1]   — bool mask, True = already visited
        lengths:       [batch]        — accumulated travel distance
        step:          int            — current step count
    """

    def __init__(self, coords, demands, capacity, cur_node, used_capacity, visited, lengths, step):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.num_nodes = coords.size(1)
        self.cur_node = cur_node
        self.used_capacity = used_capacity
        self.visited = visited
        self.lengths = lengths
        self.step = step

    @staticmethod
    def initialize(coords, demands, capacity):
        """
        Create initial state: all vehicles start at depot (index 0).

        Args:
            coords: [batch, N+1, 2]
            demands: [batch, N+1]
            capacity: float
        """
        batch_size = coords.size(0)
        num_nodes = coords.size(1)
        device = coords.device

        cur_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        used_capacity = torch.zeros(batch_size, device=device)
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        # Depot is not marked visited — vehicles can revisit it
        lengths = torch.zeros(batch_size, device=device)

        return StateCVRP(coords, demands, capacity, cur_node, used_capacity, visited, lengths, step=0)

    def update(self, selected):
        """
        Move vehicles to selected nodes and return new state + step costs.

        Args:
            selected: [batch] — indices of selected nodes

        Returns:
            new_state: updated StateCVRP
            step_cost: [batch] — distance traveled this step (positive)
        """
        batch_size = self.coords.size(0)
        device = self.coords.device

        # Gather coordinates
        cur_coords = self.coords[torch.arange(batch_size, device=device), self.cur_node]  # [batch, 2]
        next_coords = self.coords[torch.arange(batch_size, device=device), selected]      # [batch, 2]

        # Step distance
        step_cost = (cur_coords - next_coords).norm(dim=-1)  # [batch]

        # Update used capacity: reset if returning to depot, else add demand
        is_depot = (selected == 0)
        new_demands = self.demands[torch.arange(batch_size, device=device), selected]
        new_used_capacity = torch.where(is_depot, torch.zeros_like(self.used_capacity),
                                        self.used_capacity + new_demands)

        # Update visited mask: mark customer nodes as visited (not depot)
        new_visited = self.visited.clone()
        # Only mark non-depot nodes as visited
        customer_mask = (selected != 0)
        new_visited[torch.arange(batch_size, device=device), selected] = (
            new_visited[torch.arange(batch_size, device=device), selected] | customer_mask
        )

        new_lengths = self.lengths + step_cost

        return StateCVRP(
            self.coords, self.demands, self.capacity,
            selected, new_used_capacity, new_visited, new_lengths, self.step + 1
        ), step_cost

    def get_feasible_mask(self):
        """
        Returns mask where True = INFEASIBLE (cannot select).

        Rules:
            - Already visited customers are infeasible
            - Customers whose demand exceeds remaining capacity are infeasible
            - Depot (index 0) is always feasible (vehicle can return)
            - If currently at depot and all customers visited, everything is infeasible
              except depot (to allow episode termination check)

        Returns:
            mask: [batch, N+1] bool tensor, True = infeasible
        """
        batch_size = self.coords.size(0)
        device = self.coords.device

        remaining_cap = self.capacity - self.used_capacity  # [batch]

        # Start with visited mask
        mask = self.visited.clone()

        # Also mask customers whose demand exceeds remaining capacity
        exceeds_cap = self.demands > remaining_cap.unsqueeze(-1)  # [batch, N+1]
        mask = mask | exceeds_cap

        # Depot is always feasible
        mask[:, 0] = False

        return mask

    def all_finished(self):
        """True when all customer nodes (indices 1..N) have been visited in every batch element."""
        return self.visited[:, 1:].all()
