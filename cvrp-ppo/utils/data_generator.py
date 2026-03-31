import torch
import pickle
import os
from typing import Optional, Tuple


# Capacity matched to graph size — Kool et al. 2019 protocol (§6)
CAPACITY_MAP = {10: 20, 20: 30, 50: 40, 100: 50}


def get_capacity(graph_size: int) -> int:
    """Return vehicle capacity for a given graph size."""
    if graph_size not in CAPACITY_MAP:
        raise ValueError(f"Unsupported graph_size={graph_size}. Must be one of {list(CAPACITY_MAP.keys())}")
    return CAPACITY_MAP[graph_size]


def generate_instances(
    num_samples: int,
    graph_size: int,
    capacity: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Generate CVRP instances following Kool et al. (2019) protocol.

    Args:
        num_samples: batch size B
        graph_size:  number of customers N
        capacity:    vehicle capacity Q (auto-detected from graph_size if None)
        device:      torch device string — tensors are created on this device

    Returns:
        coords:   [B, N+1, 2] — depot at index 0, customers at 1..N
        demands:  [B, N+1]    — depot demand=0, customer demands in {1,...,9}
        capacity: int         — vehicle capacity Q
    """
    if capacity is None:
        capacity = get_capacity(graph_size)

    # Depot + customer coordinates ~ U(0,1)^2
    coords = torch.FloatTensor(num_samples, graph_size + 1, 2).uniform_(0, 1).to(device)

    # Demands: depot=0, customers ~ DiscreteUniform(1, 9)
    demands = torch.zeros(num_samples, graph_size + 1, device=device)
    demands[:, 1:] = torch.randint(1, 10, (num_samples, graph_size), device=device).float()

    return coords, demands, capacity


def generate_validation_set(
    num_samples: int,
    graph_size: int,
    capacity: Optional[int] = None,
    seed: int = 12345,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Generate a fixed validation set with a deterministic seed."""
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    coords, demands, cap = generate_instances(num_samples, graph_size, capacity, device=device)
    torch.random.set_rng_state(state)
    return coords, demands, cap


def save_dataset(
    graph_size: int,
    num_samples: int,
    capacity: Optional[int] = None,
    path: str = "datasets",
    seed: int = 1234,
    filename: Optional[str] = None,
) -> str:
    """
    Generate a fixed dataset and save it to disk as a .pkl file.

    Args:
        graph_size:   number of customers N
        num_samples:  batch size B
        capacity:     vehicle capacity Q (auto-detected from graph_size if None)
        path:         directory to save the file
        seed:         torch.manual_seed for reproducibility
        filename:     custom filename (auto-generated if None)

    Returns:
        filepath: full path to the saved file
    """
    coords, demands, cap = generate_validation_set(
        num_samples, graph_size, capacity=capacity, seed=seed, device="cpu"
    )

    os.makedirs(path, exist_ok=True)

    if filename is None:
        filename = f"n{graph_size}_B{num_samples}_seed{seed}.pkl"
    filepath = os.path.join(path, filename)

    data = {
        "coords": coords,       # [B, N+1, 2]
        "demands": demands,      # [B, N+1]
        "capacity": cap,         # int
        "graph_size": graph_size,
        "num_samples": num_samples,
        "seed": seed,
    }

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved {filepath}  ({size_mb:.2f} MB)")
    return filepath


def load_dataset(
    filepath: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load a saved dataset from a .pkl file.

    Args:
        filepath: path to the .pkl file
        device:   torch device to move tensors to

    Returns:
        coords:   [B, N+1, 2]
        demands:  [B, N+1]
        capacity: int
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    coords = data["coords"].to(device)    # [B, N+1, 2]
    demands = data["demands"].to(device)   # [B, N+1]
    capacity = data["capacity"]            # int

    return coords, demands, capacity
