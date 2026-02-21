"""Kalvin - PyTorch development project with MPS support."""

__version__ = "0.1.0"

from kalvin.device import get_device, DeviceInfo
from kalvin.kalvin import Kalvin

__all__ = ["get_device", "DeviceInfo", "Kalvin", "__version__"]
