"""Dialogue-driven training (spec: ``@specs/dialogue-driven-training.md``,
``@specs/peer-dialogue.md``).

A lesson is driven by an authored **dialogue table** (``script`` + ordered
``turns``) between Trainer (T) and Trainee (K). This package implements the
configuration-time **decoder** (table → flat ordered ``list[DecodedTurn]``) and
the peer **runner** — a coverage-tracking subscriber over the harness
:class:`~training.harness.bus.MessageBus` that drives the two actors to
completion.

The decoder is single-stage and runs once at configuration time; the runner
never touches ``script`` again. Both default actors are table-reading doubles,
structurally symmetric and individually replaceable by a real trainer or trainee.
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
from training.dialogue.peer_runner import (
    PeerDivergence,
    PeerRunner,
    PeerRunResult,
    run_peer,
)
from training.dialogue.runner import (
    TableTrainee,
    TableTrainer,
)

__all__ = [
    "BAND_TO_SIG",
    "DECODEDTurn",
    "DecodedTurn",
    "DialogueTable",
    "PeerDivergence",
    "PeerRunResult",
    "PeerRunner",
    "TableTrainee",
    "TableTrainer",
    "Turn",
    "decode",
    "load_table",
    "run_peer",
]
