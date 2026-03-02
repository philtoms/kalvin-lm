# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Install with dev tools
uv sync --extra dev

# Run tests
uv run pytest

# Run single test file
uv run pytest tests/test_model.py

# Run single test
uv run pytest tests/test_model.py::test_kline_create

# Format code
uv run ruff format .

# Type check
uv run mypy src/

# Run chat TUI
python -m ui.chat
```

## Architecture

Kalvin is a knowledge graph model with BPE tokenization for text encoding and similarity matching.

### Core Components (`src/kalvin/`)

- **KLine** (`model.py`): Fundamental unit with a 64-bit `s_key` (significance key) and list of child `nodes` (token IDs). Supports bitwise significance matching via `signifies(query)`.

- **Model** (`model.py`): Collection of KLines with O(1) lookup by `s_key`. Key operations:
  - `query(sig)`: Returns KLines where `s_key & sig != 0` (bitwise AND match)
  - `expand(klines, depth)`: Recursively expands KLines and their descendants
  - Uses dual-stream generators (fast/slow) for concurrent processing

- **Kalvin** (`kalvin.py`): High-level API combining Model + Tokenizer + grammar dictionary. Key operations:
  - `encode(text)`: Tokenizes text, looks up NLP types from dictionary, creates KLines
  - `decode(sig)`: Finds KLine by significance key, decodes tokens back to text
  - `prune(level)`: Removes low-activity entries
  - Serialization: `save()`/`load()` supports binary and JSON formats

- **Tokenizer** (`tokenizer.py`): Wrapper supporting rustbpe (training) and tiktoken (inference). Trained tokenizers saved to `data/tokenizer/`.

- **Significance** (`significance.py`): 64-bit ranking system for KLine matching:
  - S1 (bit 56): Prefix match indicator
  - S2 (bits 40-55): Partial positional match
  - S3 (bits 16-39): Unordered/generational match
  - S4 (0): No match
  - Higher = more significant; comparable as integers

### Data Dependencies

- Grammar dictionary: `/Volumes/USB-Backup/ai/data/tidy-ts/simplestories-1_grammar.json` - maps token IDs to NLP types
- Default tokenizer: `data/tokenizer/tokenizer-32768`

### UI (`ui/chat/`)

Textual TUI application with modular structure:
- `app.py` - Main KalvinApp
- `dialogs.py` - FileLoadDialog with mount point navigation
- `regions/config.py` - Model/grammar path configuration
- `regions/chat.py` - Chat input and response display

## Key Patterns

- Activity tracking: `Counter` tracks token usage frequency for pruning decisions
- Bitwise significance: `s_key` encodes match priority in upper bits for efficient comparison
- Dual-stream iteration: `query()` and `expand()` return (fast, slow) generators for prioritized processing
