"""Dialogue-driven training (spec: ``@specs/dialogue-driven-training.md``,
``@specs/dialogue-runner.md``).

A lesson is driven by an authored **dialogue table** (``script`` + ordered
``turns``) between Trainer (T) and Trainee (K). This package implements the
configuration-time **decoder** (table → flat ordered ``list[DecodedTurn]``),
the dialogue **actors** (:mod:`training.dialogue.actors`), and the
**runner** (:mod:`training.dialogue.runner`) — a coverage-tracking subscriber
over the harness :class:`~training.harness.bus.MessageBus` that drives the two
actors to completion.

The decoder is single-stage and runs once at configuration time. Both default
actors are table-reading, structurally symmetric, and individually replaceable
by a real trainer or trainee.
"""

from training.dialogue.decoder import (
    BAND_TO_SIG,
    DecodedTurn,
    DialogueTable,
    Turn,
    decode,
    load_table,
)
from training.dialogue.actors import (
    TableTrainee,
    TableTrainer,
)
from training.dialogue.runner import (
    Divergence,
    Runner,
    RunResult,
    run,
)

__all__ = [
    "BAND_TO_SIG",
    "DecodedTurn",
    "DialogueTable",
    "Divergence",
    "RunResult",
    "Runner",
    "TableTrainee",
    "TableTrainer",
    "Turn",
    "decode",
    "load_table",
    "run",
]
