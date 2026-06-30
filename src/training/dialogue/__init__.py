"""Dialogue-driven training (spec: ``@specs/dialogue-driven-training.md``).

A lesson is driven by an authored **dialogue table** (``script`` + ordered
``turns``) between Trainer (T) and Kalvin (K). This package implements the
configuration-time **decoder** that turns a table into a flat ordered list of
``DecodedTurn`` (Phase 1 of the plan), plus the downstream stateless supply
rule, self-cursored stub, and dispatch loop (Phases 2–4).

The decoder is a single-stage, pre-loop function (spec §Decoder): symbol
resolution is deterministic and total, so it runs once at configuration time
and the training loop never touches ``script`` again.
"""

from training.dialogue.decoder import (
    BAND_TO_SIG,
    DECODEDTurn,
    DecodedTurn,
    DialogueTable,
    Turn,
    decode,
    load_table,
)
from training.dialogue.supply import (
    HeldIndex,
    SupplyMiss,
    TrainerResponse,
    build_held_index,
    opening,
    respond,
    supply,
    terminal_significance,
)
from training.dialogue.loop import (
    LoopError,
    LoopResult,
    run_session,
    run_with_held,
)
from training.dialogue.stub_kagent import StubKAgent

__all__ = [
    "BAND_TO_SIG",
    "DECODEDTurn",
    "DecodedTurn",
    "DialogueTable",
    "HeldIndex",
    "LoopError",
    "LoopResult",
    "StubKAgent",
    "SupplyMiss",
    "TrainerResponse",
    "Turn",
    "build_held_index",
    "decode",
    "load_table",
    "opening",
    "respond",
    "run_session",
    "run_with_held",
    "supply",
    "terminal_significance",
]
