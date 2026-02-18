# Kalvin

PyTorch development project with MPS (Apple Silicon) support.

## Setup

```bash
# Install dependencies
uv sync

# Install with dev tools
uv sync --extra dev

# Install with Jupyter support
uv sync --extra notebooks
```

## Usage

```python
from kalvin import get_device, DeviceInfo
from kalvin.device import print_device_status

# Check available devices
print_device_status()

# Get the best available device
device = get_device()
print(f"Using device: {device}")

# Move tensors to device
import torch
x = torch.randn(3, 3).to(device)
```

## Project Structure

```
kalvin/
├── src/kalvin/
│   ├── __init__.py      # Package exports
│   ├── device.py        # Device detection and management
│   └── utils.py         # Utility functions
├── tests/
│   └── test_device.py   # Unit tests
├── pyproject.toml       # Project configuration
└── README.md
```

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run ruff format .

# Type check
uv run mypy src/
```
