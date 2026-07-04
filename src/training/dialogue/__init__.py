"""Dialogue-driven training (spec: ``@specs/dialogue-driven-training.md``).

A lesson is driven by an authored **dialogue table** (``script`` + ordered
``turns``) between Trainer (T) and Trainee (K). This package implements the
configuration-time **decoder** (table → flat ordered ``list[DecodedTurn]``) and
the bus-agnostic **runner** that alternates the two actors to exhaustion.

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
    Actor,
    ActorDivergence,
    RunResult,
    TableTrainee,
    TableTrainer,
    default_actors,
    run,
    run_table,
)

__all__ = [
    "Actor",
    "ActorDivergence",
    "BAND_TO_SIG",
    "DECODEDTurn",
    "DecodedTurn",
    "DialogueTable",
    "PeerDivergence",
    "PeerRunResult",
    "PeerRunner",
    "RunResult",
    "TableTrainee",
    "TableTrainer",
    "Turn",
    "decode",
    "default_actors",
    "load_table",
    "run",
    "run_peer",
    "run_table",
]
