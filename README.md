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

### Device Management

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

### Kalvin Model

```python
from kalvin import Kalvin

# Initialize or load a model
kalvin = Kalvin()
kalvin = Kalvin.load("data/kalvin.bin")

# Encode text
embedding = kalvin.encode("Hello, world!")

# Save model
kalvin.save("data/kalvin.bin")
```

### Tokenizer

```python
from kalvin.tokenizer import Tokenizer

# Train a new tokenizer
tokenizer = Tokenizer()
tokenizer.train(["text data...", "more text..."], vocab_size=4096)
tokenizer.save_to_directory("data/tokenizer")

# Load a trained tokenizer
tokenizer = Tokenizer.from_directory("data/tokenizer")

# Encode/decode
ids = tokenizer.encode("Hello, world!")
text = tokenizer.decode(ids)
```

## Scripts

### encode_text.py

Encode text files into Kalvin embeddings:

```bash
# Encode a text file
uv run python scripts/encode_text.py path/to/data.txt

# Encode into an existing model
uv run python scripts/encode_text.py path/to/data.txt data/kalvin.bin
```

### train_tokenizer.py

Train a BPE tokenizer on text data. Supports `.txt`, `.json`, and `.parquet` files.

**JSON formats supported:**

- `{"summaries": [{"summary": "..."}]}`
- `[{"summary": "..."}]`

```bash
# Train from a text file/directory (default vocab_size: 4096)
uv run python scripts/train_tokenizer.py path/to/data.txt

# Train from JSON file/directory
uv run python scripts/train_tokenizer.py path/to/data.json

# Train from parquet file/directory
uv run python scripts/train_tokenizer.py path/to/parquets/

# Train with custom vocab size
uv run python scripts/train_tokenizer.py path/to/data.txt 8192

# Custom JSON field name (default: summary)
uv run python scripts/train_tokenizer.py data.json 4096 --json-field text

# Custom output location
uv run python scripts/train_tokenizer.py data.txt 4096 -o my_tokenizer -n my_tok
```

## Project Structure

```
kalvin/
├── src/kalvin/
│   ├── __init__.py      # Package exports
│   ├── device.py        # Device detection and management
│   ├── kalvin.py        # Main Kalvin model
│   ├── model.py         # Model internals
│   ├── tokenizer.py     # BPE tokenizer wrapper
│   ├── significance.py  # Significance testing
│   └── utils.py         # Utility functions
├── scripts/
│   ├── encode_text.py   # Text encoding script
│   └── train_tokenizer.py  # Tokenizer training script
├── tests/
├── pyproject.toml
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
