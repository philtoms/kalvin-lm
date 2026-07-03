"""Run a dialogue end-to-end through the table-reading trainer and trainee.

Loads a dialogue table, decodes it, drives the bus-agnostic runner
(:func:`training.dialogue.run`) with a fresh :class:`TableTrainer` and
:class:`TableTrainee`, and prints a PASS/FAIL summary. With ``--verbose`` it
traces the interleaved T/K exchange.

With ``--rationalise`` the trainee is a :class:`Rationaliser` — a real,
stateful, rationalising trainee and drop-in ``TableTrainee`` replacement —
while the trainer stays a ``TableTrainer`` (the deterministic oracle). This is
the canonical demonstration that a rationalising trainee reproduces the golden
master through the runner.

The runner is bus-agnostic: there is no harness message bus and no adapter here.
Bus integration arrives with the real actors.

Usage::

    python scripts/dialogue_run.py                             # default dialogue
    python scripts/dialogue_run.py scripts/dialogue-mhall.json # explicit path
    python scripts/dialogue_run.py --verbose                   # per-turn trace
    python scripts/dialogue_run.py --rationalise               # Rationaliser trainee

Exit code is 0 on a completing run (table exhausted), 1 on an actor divergence
or incomplete run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SYS_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SYS_SRC) not in sys.path:
    sys.path.insert(0, str(_SYS_SRC))

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4  # noqa: E402
from kalvin.nlp_tokenizer import NLPTokenizer  # noqa: E402
from kalvin.signifier import NLPSignifier  # noqa: E402
from training.dialogue import (  # noqa: E402
    ActorDivergence,
    TableTrainer,
    decode,
    load_table,
    run,
)
from training.dialogue.rationalise import Rationaliser  # noqa: E402

_SIG_TO_BAND = {
    SIG_S1: "S1",
    SIG_S2: "S2",
    SIG_S3: "S3",
    SIG_S4: "S4",
}

DEFAULT_DIALOGUE = "scripts/dialogue-mhall.json"


def _label_of(kv) -> str:
    """Best-effort symbolic label for a KValue's kline (for trace output)."""
    dbg = kv.kline.dbg
    if dbg is None:
        return f"sig=0x{kv.kline.signature:x}"
    return dbg.label or dbg.decoded or f"sig=0x{kv.kline.signature:x}"


def _trace(events: list, decoded) -> str:
    """A per-turn interleaved trace of the exchange, tagged by actor from the table."""
    # Reconstruct the actor/op tag per emitted event from the decoded order.
    tags: list[str] = []
    di = 0
    for ev in events:
        while di < len(decoded) and decoded[di].value.kline != ev.proposal.kline:
            di += 1
        if di < len(decoded):
            tags.append(f"{decoded[di].role} {decoded[di].op}")
            di += 1
        else:  # pragma: no cover - defensive
            tags.append("? ?")

    lines = []
    for i, (ev, tag) in enumerate(zip(events, tags)):
        actor, op = tag.split(" ", 1)
        band = _SIG_TO_BAND.get(ev.proposal.significance, "?")
        lines.append(
            f"  {actor}#{i:2} {op:13} {_label_of(ev.proposal):10} {band}"
        )
    return "\n".join(lines)


def _summary(result, decoded, dialogue_path: str, verbose: bool, trainee_kind: str) -> str:
    n_t = sum(1 for t in decoded if t.role == "T")
    n_k = sum(1 for t in decoded if t.role == "K")
    header = (
        f"Dialogue session: {dialogue_path}\n"
        f"  trainer             : TableTrainer\n"
        f"  trainee             : {trainee_kind}\n"
        f"  table trainer turns : {n_t}\n"
        f"  table trainee turns : {n_k}\n"
        f"  events emitted      : {len(result.events)}\n"
        f"  complete            : {result.complete}\n"
    )
    if verbose:
        return header + "\nExchange:\n" + _trace(result.events, decoded)
    return header


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a dialogue end-to-end through the table-reading actors."
    )
    parser.add_argument(
        "dialogue",
        nargs="?",
        default=DEFAULT_DIALOGUE,
        help=f"Path to a dialogue JSON (default: {DEFAULT_DIALOGUE})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the exchange turn by turn.",
    )
    parser.add_argument(
        "--rationalise",
        action="store_true",
        help=(
            "Substitute a Rationaliser (a real, stateful rationalising trainee) "
            "for the default TableTrainee. The trainer stays a TableTrainer "
            "(the deterministic oracle)."
        ),
    )
    args = parser.parse_args(argv)

    dialogue_path = args.dialogue
    tok = NLPTokenizer()
    sigf = NLPSignifier()

    table = load_table(json.loads(Path(dialogue_path).read_text()))
    decoded = decode(table, tokenizer=tok, signifier=sigf)

    trainer = TableTrainer(decoded)
    if args.rationalise:
        trainee = Rationaliser(sigf)
        trainee_kind = "Rationaliser"
    else:
        from training.dialogue import TableTrainee

        trainee = TableTrainee(decoded)
        trainee_kind = "TableTrainee"

    try:
        result = run(decoded, trainer=trainer, trainee=trainee)
    except ActorDivergence as exc:
        print(
            f"FAIL — {exc.role} divergence at cursor {exc.cursor}: "
            f"expected {_label_of(exc.expected)} "
            f"(sig={exc.expected.significance:#x}), "
            f"emitted {_label_of(exc.emitted)} "
            f"(sig={exc.emitted.significance:#x})",
            file=sys.stderr,
        )
        return 1

    print(_summary(result, decoded, dialogue_path, args.verbose, trainee_kind))
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
