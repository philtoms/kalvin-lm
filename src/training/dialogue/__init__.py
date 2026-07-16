"""Dialogue-driven training.

An authored **dialogue table** drives a lesson between Trainer (T) and
Trainee (K). The package has a configuration-time **decoder** (table → flat
ordered ``list[DecodedTurn]``), dialogue **actors**, and a **runner** that
drives the two actors over the harness ``MessageBus`` and tracks how much of
the authored exchange they traverse.

The decoder runs once at configuration time. The default actors are
table-reading scaffolding, individually replaceable by a real trainer or
trainee.
"""

from training.dialogue.decoder import (
    BAND_TO_SIG,
    DecodedTurn,
    DialogueScript,
    Turn,
    decode,
    decode_events,
    load_script,
)
from training.dialogue.actors import (
    ScriptTrainee,
    ScriptTrainer,
)
from training.dialogue.runner import (
    Divergence,
    GroundingDivergence,
    Runner,
    RunResult,
    run,
)

__all__ = [
    "BAND_TO_SIG",
    "DecodedTurn",
    "DialogueScript",
    "Divergence",
    "GroundingDivergence",
    "RunResult",
    "Runner",
    "ScriptTrainee",
    "ScriptTrainer",
    "Turn",
    "decode",
    "decode_events",
    "load_script",
    "run",
]
