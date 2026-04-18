import torch
import numpy as np
import random


def set_seed(seed: int):
    """Set random seed for reproducibility across torch, numpy, and python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # FIXED: CLAUDE.md requires deterministic cuDNN for reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
