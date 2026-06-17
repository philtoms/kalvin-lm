"""Centralised data-directory resolution.

All code that references the ``data/`` directory should resolve paths through
this module rather than hard-coding ``"data/…"``.  The resolution order is:

1. ``KALVIN_DATA_DIR`` environment variable (absolute path).
2. ``data/`` relative to the project root (the repo checkout that contains
   ``src/kalvin/``).

Worktrees can set ``KALVIN_DATA_DIR`` to point at the main checkout's data
directory — no symlinks needed.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def data_dir() -> Path:
    """Return the resolved data directory (creates nothing on disk)."""
    env = os.environ.get("KALVIN_DATA_DIR")
    if env:
        return Path(env)
    return _PROJECT_ROOT / "data"


# Convenience constants

TOKENIZER_DIR: str = "tokenizer"
AGENT_BIN: str = "agent.bin"


def tokenizer_dir() -> Path:
    """``<data>/tokenizer``"""
    return data_dir() / TOKENIZER_DIR


def agent_bin() -> Path:
    """``<data>/agent.bin``"""
    return data_dir() / AGENT_BIN
