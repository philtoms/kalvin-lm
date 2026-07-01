"""Run a dialogue-driven training session end-to-end against the StubKAgent.

This is the new-model replacement for the retired ``scripts/dialogue_runner.py``
(legend format + replay driver; removed in fdd177b). It loads a dialogue table,
pre-decodes it, wires the self-cursored :class:`StubKAgent` through the real
:class:`KAgentAdapter` + :class:`MessageBus`, drives the dispatch loop
(:func:`run_session`), and prints a PASS/FAIL summary. It is the runnable proof
that the stub is exercised outside the test suite, and the tool to reach for to
watch a dialogue run.

The run is synchronous (the stub resolves everything inside ``rationalise()``):
the loop submits each Trainer turn via the bus (action ``rationalise``) and
drains before the next, observing the stub's K emissions in order. Termination is
dual cursor exhaustion; the closing K (the primary's S1 countersign) is verified
by construction, not detected.

Usage::

    python scripts/dialogue_run.py                             # default dialogue
    python scripts/dialogue_run.py scripts/dialogue-mhall.json # explicit path
    python scripts/dialogue_run.py --verbose                   # per-turn trace

Exit code is 0 on a completing run (dual exhaustion reached, closing K verified),
1 on any two-sided validation failure or exhaustion violation.
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
    LoopError,
    LoopResult,
    StubKAgent,
    decode,
    load_table,
    run_session,
)
from training.harness.adapter import KAgentAdapter  # noqa: E402
from training.harness.bus import MessageBus  # noqa: E402
from training.harness.constants import TRAINEE_ROLE  # noqa: E402

# Band constants are uint64 inverted-distance values; map back to labels for
# readable output (the same mapping the decoder uses, inverted).
_SIG_TO_BAND = {
    SIG_S1: "S1",
    SIG_S2: "S2",
    SIG_S3: "S3",
    SIG_S4: "S4",
}

DEFAULT_DIALOGUE = "scripts/dialogue-mhall.json"


def _label_of(turn) -> str:
    """Best-effort symbolic label for a decoded turn's kline (for trace output)."""
    dbg = turn.value.kline.dbg
    if dbg is None:
        return f"sig=0x{turn.value.kline.signature:x}"
    return dbg.label or dbg.decoded or f"sig=0x{turn.value.kline.signature:x}"


def _trace(result: LoopResult) -> str:
    """A per-turn interleaved trace of the validated T/K exchange."""
    lines = []
    n = max(len(result.t_submissions), len(result.k_emissions))
    for i in range(n):
        if i < len(result.t_submissions):
            t = result.t_submissions[i]
            band = _SIG_TO_BAND.get(t.value.significance, "?")
            lines.append(
                f"  T#{i:2} {t.op:13} {_label_of(t):10} {band}  sig=0x{t.value.kline.signature:x}"
            )
        if i < len(result.k_emissions):
            k = result.k_emissions[i]
            band = _SIG_TO_BAND.get(k.value.significance, "?")
            lines.append(
                f"  K#{i:2} {k.op:13} {_label_of(k):10} {band}  sig=0x{k.value.kline.signature:x}"
            )
    return "\n".join(lines)


def _summary(result: LoopResult, dialogue_path: str, verbose: bool) -> str:
    closing = result.closing
    closing_desc = (
        f"{closing.op} {_label_of(closing)} @ {_SIG_TO_BAND.get(closing.value.significance, '?')}"
        if closing is not None
        else "(none — run did not close)"
    )
    header = (
        f"Dialogue session: {dialogue_path}\n"
        f"  trainer submissions : {len(result.t_submissions)}\n"
        f"  stub emissions      : {len(result.k_emissions)}\n"
        f"  dual exhaustion     : {result.dual_exhaustion}\n"
        f"  closing K           : {closing_desc}\n"
    )
    if verbose:
        return header + "\nValidated exchange:\n" + _trace(result)
    return header


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a dialogue end-to-end against the StubKAgent."
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
        help="Print the validated T/K exchange turn by turn.",
    )
    args = parser.parse_args(argv)

    dialogue_path = args.dialogue
    tok = NLPTokenizer()
    sigf = NLPSignifier()

    table = load_table(json.loads(Path(dialogue_path).read_text()))
    # Pre-decode once to get the K-rows for the stub (run_session re-decodes
    # internally too; this mirrors the spec's "stub constructed with its K-rows"
    # wiring and gives an early decode-error if the table is malformed).
    decoded = decode(table, tokenizer=tok, signifier=sigf)
    k_rows = [t for t in decoded if t.actor == "K"]

    bus = MessageBus()
    adapter = KAgentAdapter(bus, role=TRAINEE_ROLE, tokenizer=tok, signifier=sigf)
    stub = StubKAgent(adapter, k_rows)
    adapter.bind(stub)

    try:
        result = run_session(table, adapter=adapter, tokenizer=tok, signifier=sigf)
    except LoopError as exc:
        print(f"FAIL — {exc.kind}: {exc}", file=sys.stderr)
        return 1

    print(_summary(result, dialogue_path, args.verbose))
    return 0 if result.dual_exhaustion else 1


if __name__ == "__main__":
    raise SystemExit(main())
