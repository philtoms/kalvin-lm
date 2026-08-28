"""Dialogue-driven training.

An authored **dialogue table** drives a lesson between Trainer (T) and
Trainee (K). The package has a configuration-time **decoder** (table → flat
ordered ``list[DecodedTurn]``), the rationalising **engine** and its
**harness** that drive a lesson over compiled ``.ks`` scripts.
"""

from dialogue.decoder import (
    BAND_TO_SIG,
    DecodedTurn,
    DialogueScript,
    Turn,
    decode,
    decode_events,
    load_script,
    load_script_file,
)

__all__ = [
    "BAND_TO_SIG",
    "DecodedTurn",
    "DialogueScript",
    "Turn",
    "decode",
    "decode_events",
    "load_script",
    "load_script_file",
]
