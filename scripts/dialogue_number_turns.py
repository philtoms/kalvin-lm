"""Add ``turn`` and ``role_turn`` indices to every turn across a dialogue chain.

A dialogue table's ``turns`` are an ordered list, and a table may declare
``priors`` — other dialogue files whose turns run before its own (resolved
recursively; see the decoder's ``load_table``). This script numbers the turns
across the **whole resolved chain** and writes the indices back into each
originating file:

- ``turn`` — the turn's 1-indexed position across the full prior chain (the
  root table's own turns continue the count begun in its priors).
- ``role_turn`` — the turn's 1-indexed position among its own ``role``
  (``"T"`` or ``"K"``), accumulated across the chain.

So if a prior's last turn is ``turn 34`` (``K``, ``role_turn 18``) and the root
table's first own turn is a ``T``, that turn is ``turn 35, role_turn 17``
(continuing the prior's last ``T`` count). Both keys are overwritten if already
present, so the script is idempotent. Files are rewritten in place; pass
``--check`` to print the resolved chain's numbering instead of writing.

The added keys are ignored by the decoder, so an annotated table loads
unchanged. A file may appear as a prior to several tables; numbering is
recomputed per invocation against the chain rooted at the file you pass, so
re-annotate each root whose chain you want consistent.

Usage::

    python scripts/dialogue_number_turns.py scripts/dialogue-wdmh.json
    python scripts/dialogue_number_turns.py scripts/dialogue-mhall.json   # no priors
    python scripts/dialogue_number_turns.py scripts/dialogue-wdmh.json --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve_chain(path: Path, *, seen: set[Path]) -> list[Path]:
    """The resolved prior chain for ``path`` in decode order.

    Mirrors the decoder's ``load_table``: each prior is loaded recursively
    (its own priors first), then appended in list order; the root file's own
    turns come last. Returns the ordered list of file paths whose turns
    compose the chain, root last. Cycle-guarded via ``seen`` (a cycle is a
    malformed table).
    """
    path = path.resolve()
    if path in seen:
        raise ValueError(f"prior cycle detected at {path}")
    seen.add(path)
    table = json.loads(path.read_text())
    ordered: list[Path] = []
    for prior in table.get("priors", []) or []:
        ordered.extend(_resolve_chain((Path.cwd() / prior), seen=seen))
    ordered.append(path)
    return ordered


def _annotate_chain(chain: list[Path]) -> dict[Path, list[dict]]:
    """Number every turn across ``chain``; return ``{path: annotated turns}``.

    Walks the chain in order, accumulating a global ``turn`` counter and a
    per-role ``role_turn`` counter, and stamps each turn (in its origin file's
    list) with both. Returns each path's own annotated turns so the caller can
    rewrite each file independently.
    """
    global_turn = 0
    per_role: dict[str, int] = {"T": 0, "K": 0}
    out: dict[Path, list[dict]] = {}
    for path in chain:
        table = json.loads(path.read_text())
        turns = table.get("turns")
        if not isinstance(turns, list):
            raise ValueError(f"{path}: no 'turns' list found")
        annotated: list[dict] = []
        for turn in turns:
            global_turn += 1
            role = turn.get("role")
            if role not in ("T", "K"):
                raise ValueError(
                    f"{path} turn {global_turn}: role must be 'T' or 'K', got {role!r}"
                )
            per_role[role] += 1
            annotated.append(_stamp(turn, global_turn, per_role[role]))
        out[path] = annotated
    return out


def _stamp(turn: dict, turn_no: int, role_no: int) -> dict:
    """Return ``turn`` with ``turn``/``role_turn`` set after ``role``.

    Drops any stale ``turn``/``role_turn`` so the fresh values land right after
    ``role`` (not wherever a previous run left them). ``role`` is required —
    the decoder rejects a roleless turn too.
    """
    stamped: dict = {}
    for key, value in turn.items():
        if key in ("turn", "role_turn"):
            continue
        stamped[key] = value
        if key == "role":
            stamped["turn"] = turn_no
            stamped["role_turn"] = role_no
    if "role" not in stamped:
        raise ValueError(f"turn {turn_no}: missing 'role'")
    return stamped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add `turn` and `role_turn` indices to each turn across a dialogue "
            "chain (priors accumulate)."
        )
    )
    parser.add_argument(
        "dialogue",
        help="Path to a dialogue JSON (the chain root; rewritten in place "
        "along with its priors unless --check).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the resolved chain's numbering to stdout instead of writing.",
    )
    args = parser.parse_args(argv)

    root = Path(args.dialogue)
    try:
        chain = _resolve_chain(root, seen=set())
        annotated = _annotate_chain(chain)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    if args.check:
        for path in chain:
            print(f"# {path}")
            for turn in annotated[path]:
                role = turn.get("role", "?")
                print(
                    f"  {turn['turn']:3} {role} {turn['role_turn']:3} "
                    f"{turn.get('op', '(annot)')}"
                )
        return 0

    for path in chain:
        table = json.loads(path.read_text())
        table["turns"] = annotated[path]
        path.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
