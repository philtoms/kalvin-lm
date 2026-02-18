"""Utility functions for PyTorch development."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(num: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    for suffix, divisor in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if num >= divisor:
            return f"{num / divisor:.1f}{suffix}"
    return str(num)


def get_memory_usage(device: Optional[torch.device] = None) -> dict[str, float]:
    """
    Get current memory usage for the specified device.

    Args:
        device: Target device (defaults to best available).

    Returns:
        Dictionary with memory statistics in GB.
    """
    if device is None:
        from kalvin.device import get_device

        device = get_device()

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return {"allocated_gb": allocated, "reserved_gb": reserved}
    elif device.type == "mps":
        # MPS doesn't have direct memory query, use system memory as approximation
        import psutil

        mem = psutil.virtual_memory()
        return {
            "system_used_gb": mem.used / (1024**3),
            "system_total_gb": mem.total / (1024**3),
        }
    return {"info": "CPU memory tracking not available"}
