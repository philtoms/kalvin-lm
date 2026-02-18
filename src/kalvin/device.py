"""Device detection and management utilities for PyTorch."""

from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    """Information about the available compute device."""

    device_type: str
    device_name: str
    is_available: bool
    memory_gb: float | None = None

    def __str__(self) -> str:
        status = "✓" if self.is_available else "✗"
        mem_info = f" ({self.memory_gb:.1f} GB)" if self.memory_gb else ""
        return f"[{status}] {self.device_type}: {self.device_name}{mem_info}"


def get_device() -> torch.device:
    """
    Get the best available compute device.

    Priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> list[DeviceInfo]:
    """
    Get information about all available compute devices.

    Returns:
        List of DeviceInfo objects for each device type.
    """
    devices = []

    # CUDA
    cuda_available = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    cuda_memory = None
    if cuda_available:
        cuda_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    devices.append(DeviceInfo("CUDA", cuda_name, cuda_available, cuda_memory))

    # MPS (Apple Silicon)
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    mps_name = "Apple Silicon" if mps_available else "Not available"
    if mps_available and not mps_built:
        mps_name = "Built but not available"
    devices.append(DeviceInfo("MPS", mps_name, mps_available))

    # CPU (always available)
    devices.append(DeviceInfo("CPU", "Generic", True))

    return devices


def print_device_status() -> None:
    """Print the status of all available devices."""
    print("Device Status:")
    print("-" * 50)
    for info in get_device_info():
        print(f"  {info}")
    print("-" * 50)
    print(f"Selected device: {get_device()}")


def to_device(data: object, device: torch.device | None = None) -> object:
    """
    Move tensors or models to the specified device.

    Args:
        data: Tensor, model, or dict/list of tensors.
        device: Target device (defaults to best available).

    Returns:
        Data on the target device.
    """
    if device is None:
        device = get_device()

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif hasattr(data, "to"):
        return data.to(device)
    return data
