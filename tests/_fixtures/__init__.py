"""Stable in-test copies of dialogue fixtures.

``scripts/dialogue-mhall.json`` is a user-editable artefact (the reference
dialogue for "Mary had a little lamb"). Tests must not depend on its current
contents, so the canonical acceptance fixture is frozen here as a Python dict.
``mhall_table()`` returns a fresh copy each call so a test can mutate it
without affecting other tests.
"""

from __future__ import annotations

import copy
import json

# The "Mary had a little lamb" reference dialogue. Frozen from
# ``scripts/dialogue-mhall.json``; if the script's structure is intentionally
# changed, update this constant deliberately — tests assert against it.
MHALL_SCRIPT = (
    "(Mary had a little lamb)\n"
    "MHALL == SVO =>\n"
    "   S(ubject) = M\n"
    "   V(erb) = H\n"
    "   O(bject) = ALL =>\n"
    "     A > D(et)\n"
    "     L > M(od)\n"
    "     L > O"
)

MHALL_TURNS = [
    {"nodes": ["SVO"], "op": "COUNTERSIGNS", "role": "T", "signature": "MHALL", "significance": "S2"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "MHALL", "significance": "S4"},
    {"nodes": ["Mary", "had", "a", "little", "lamb"], "op": "CANONIZES", "role": "T", "signature": "MHALL", "significance": "S2"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "Mary", "significance": "S4"},
    {"nodes": ["M", "ary"], "op": "CANONIZES", "role": "T", "signature": "Mary", "significance": "S1"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "had", "significance": "S4"},
    {"nodes": ["h", "ad"], "op": "CANONIZES", "role": "T", "signature": "had", "significance": "S1"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "a", "significance": "S4"},
    {"nodes": ["Det"], "op": "CONNOTES", "role": "T", "signature": "a", "significance": "S2"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "Det", "significance": "S4"},
    {"nodes": ["D", "et"], "op": "CANONIZES", "role": "T", "signature": "Det", "significance": "S1"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "little", "significance": "S4"},
    {"nodes": ["l", "ittle"], "op": "CANONIZES", "role": "T", "signature": "little", "significance": "S1"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "lamb", "significance": "S4"},
    {"nodes": ["l", "amb"], "op": "CANONIZES", "role": "T", "signature": "lamb", "significance": "S1"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "SVO", "significance": "S4"},
    {"nodes": ["Subject", "Verb", "Object"], "op": "CANONIZES", "role": "T", "signature": "SVO", "significance": "S2"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "Subject", "significance": "S4"},
    {"nodes": ["Sub", "ject"], "op": "CANONIZES", "role": "T", "signature": "Subject", "significance": "S1"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "Verb", "significance": "S4"},
    {"nodes": ["V", "er", "b"], "op": "CANONIZES", "role": "T", "signature": "Verb", "significance": "S1"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "Object", "significance": "S4"},
    {"nodes": ["Ob", "ject"], "op": "CANONIZES", "role": "T", "signature": "Object", "significance": "S1"},
    {"role": "K"},
    {"nodes": ["Subject"], "op": "CONNOTES", "role": "K", "signature": "Mary", "significance": "S3"},
    {"nodes": ["Subject"], "op": "DENOTES", "role": "T", "signature": "Mary", "significance": "S1"},
    {"nodes": ["Verb"], "op": "CONNOTES", "role": "K", "signature": "had", "significance": "S3"},
    {"nodes": ["Verb"], "op": "DENOTES", "role": "T", "signature": "had", "significance": "S1"},
    {"nodes": ["a", "little", "lamb"], "op": "CANONIZES", "role": "K", "signature": "ALL", "significance": "S2"},
    {"nodes": ["a", "little", "lamb"], "op": "CANONIZES", "role": "T", "signature": "ALL", "significance": "S2"},
    {"nodes": [], "op": "IDENTITY", "role": "K", "signature": "a", "significance": "S4"},
    {"nodes": ["Det"], "op": "CONNOTES", "role": "T", "signature": "a", "significance": "S2"},
    {"role": "K"},
    {"nodes": ["SVO"], "op": "COUNTERSIGNS", "role": "K", "signature": "MHALL", "significance": "S1"},
]


def mhall_table() -> dict:
    """Return a fresh deep copy of the MHALL dialogue-table dict.

    Each call returns an independent copy so a test can mutate the table
    (drop turns, rewrite fields) without leaking state to other tests.
    """
    return copy.deepcopy({"script": MHALL_SCRIPT, "turns": MHALL_TURNS})


def mhall_json() -> str:
    """The MHALL table serialised as JSON (mirrors the on-disk file shape)."""
    return json.dumps(mhall_table())
