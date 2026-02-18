"""Tests for device detection and management."""

import torch

from kalvin.device import DeviceInfo, get_device, get_device_info


def test_get_device_returns_torch_device():
    """Test that get_device returns a valid torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")


def test_get_device_info_returns_list():
    """Test that get_device_info returns a list of DeviceInfo."""
    devices = get_device_info()
    assert isinstance(devices, list)
    assert len(devices) == 3  # CUDA, MPS, CPU


def test_cpu_always_available():
    """Test that CPU is always reported as available."""
    devices = get_device_info()
    cpu_device = next(d for d in devices if d.device_type == "CPU")
    assert cpu_device.is_available is True


def test_device_info_str():
    """Test DeviceInfo string representation."""
    info = DeviceInfo("Test", "TestDevice", True, 8.0)
    assert "Test" in str(info)
    assert "TestDevice" in str(info)
    assert "8.0" in str(info)
