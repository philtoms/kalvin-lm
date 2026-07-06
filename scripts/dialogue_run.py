"""Run a dialogue end-to-end through the table-reading trainer and trainee.

Loads a dialogue table, decodes it, and drives it to completion. The **table
selects the run regime**: a table with no ``peer`` section drives the
bus-agnostic ordered runner (:func:`training.dialogue.run`) with a fresh
:class:`TableTrainer` and :class:`TableTrainee`; a table carrying a ``peer``
section drives the sink-shaped :class:`PeerRunner`, bridging the table-reading
actors onto it. There are no CLI flags for the regime or peer modifiers —
they live in the table. Prints a PASS/FAIL summary; with ``--verbose`` it
traces the exchange.

For an ordered table, ``--rationalise`` substitutes a
:class:`RationalisingTrainee` (a real, stateful, rationalising trainee) for the
default ``TableTrainee``, and ``--synthesize`` substitutes a
:class:`SynthesizingTrainer` (a real trainer that derives each turn from the
compiled script) for the default ``TableTrainer``. The two flags are
orthogonal: passing both runs the two real actors against the same golden
master. They apply in both regimes — the real actors are drop-in (adapter-
driven), so a peer table with ``--rationalise`` / ``--synthesize`` runs the
real actors through the bus-subscriber relay. Note real actors do not
reproduce the table exactly, so peer mode with them typically needs
``on_divergence: "accept"`` in the table's ``peer`` section (else off-table
emissions raise ``PeerDivergence``).

The runner is bus-agnostic: there is no harness message bus and no adapter
here. Bus integration arrives with the real actors.

Usage::

    python scripts/dialogue_run.py                             # default dialogue
    python scripts/dialogue_run.py scripts/dialogue-mhall.json # explicit path
    python scripts/dialogue_run.py --verbose                   # per-turn trace
    python scripts/dialogue_run.py --rationalise               # RationalisingTrainee
    python scripts/dialogue_run.py --synthesize                # SynthesizingTrainer
    python scripts/dialogue_run.py --synthesize --rationalise  # both real actors
    # A peer-mode table (one with a "peer" section) runs through PeerRunner.

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
from ks.compiler import compile_source  # noqa: E402
from training.dialogue import (  # noqa: E402
    ActorDivergence,
    PeerDivergence,
    TableTrainee,
    TableTrainer,
    decode,
    load_table,
    run,
    run_peer,
)
from training.dialogue.runner import (  # noqa: E402
    Actor,
    RationalisingTrainee,
    SynthesizingTrainer,
)

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


def _summary(
    result,
    decoded,
    dialogue_path: str,
    verbose: bool,
    trainer_kind: str,
    trainee_kind: str,
) -> str:
    n_t = sum(1 for t in decoded if t.role == "T")
    n_k = sum(1 for t in decoded if t.role == "K")
    header = (
        f"Dialogue session: {dialogue_path}\n"
        f"  trainer             : {trainer_kind}\n"
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
            "Substitute a RationalisingTrainee (a real, stateful rationalising trainee) "
            "for the default TableTrainee. The trainer stays a TableTrainer "
            "(the deterministic oracle)."
        ),
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help=(
            "Substitute a SynthesizingTrainer (a real trainer that derives each "
            "turn from the compiled script) for the default TableTrainer. "
            "Orthogonal to --rationalise."
        ),
    )
    args = parser.parse_args(argv)

    dialogue_path = args.dialogue
    tok = NLPTokenizer()
    sigf = NLPSignifier()

    table = load_table(json.loads(Path(dialogue_path).read_text()))
    decoded = decode(table, tokenizer=tok, signifier=sigf)

    # The table selects the run regime: a ``peer`` section drives the sink-
    # shaped PeerRunner (bridging the table-reading actors onto it); no ``peer``
    # section drives the synchronous ordered run. The regime is table-driven —
    # there are no CLI flags for it. Peer modifiers (on_divergence) come from
    # the peer section.
    if table.is_peer:
        assert table.peer is not None  # narrowed by is_peer
        # The --synthesize/--rationalise flags honour peer mode too: the real
        # actors are drop-in (adapter-driven). Orthogonal, as in ordered mode.
        compiled = (
            compile_source(table.script, tokenizer=tok, signifier=sigf, dev=True)
            if args.synthesize
            else None
        )
        trainer_factory = (
            (lambda sink: SynthesizingTrainer(compiled, sigf, sink=sink))
            if args.synthesize
            else (lambda sink: TableTrainer(decoded, sink=sink))
        )
        trainee_factory = (
            (lambda sink: RationalisingTrainee(sigf, sink=sink, burst_mode=True))
            if args.rationalise
            else (lambda sink: TableTrainee(decoded, sink=sink))
        )
        try:
            runner = run_peer(
                decoded,
                trainer_factory,
                trainee_factory,
                on_divergence=table.peer.on_divergence,
            )
            res = runner.run()
        except PeerDivergence as exc:
            print(
                f"FAIL — {exc.role} divergence: emitted {_label_of(exc.emitted)} "
                f"(sig={exc.emitted.significance:#x}) matches no closing or "
                f"middle content ({len(exc.unconsumed)} uncovered same-role).",
                file=sys.stderr,
            )
            return 1
        print(
            f"Peer dialogue session: {dialogue_path}\n"
            f"  on divergence       : {table.peer.on_divergence}\n"
            f"  events received     : {len(res.events)}\n"
            f"  complete            : {res.complete}\n"
            f"  covered (middle)    : {res.covered}\n"
            f"  unmatched emissions : {len(res.unmatched)}\n"
            f"  uncovered rows      : {len(res.uncovered)}"
        )
        return 0 if res.complete else 1

    if args.synthesize:
        compiled = compile_source(table.script, tokenizer=tok, signifier=sigf, dev=True)
        trainer: Actor = SynthesizingTrainer(compiled, sigf)
        trainer_kind = "SynthesizingTrainer"
    else:
        trainer = TableTrainer(decoded)
        trainer_kind = "TableTrainer"

    if args.rationalise:
        trainee: Actor = RationalisingTrainee(sigf)
        trainee_kind = "RationalisingTrainee"
    else:
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

    print(_summary(result, decoded, dialogue_path, args.verbose, trainer_kind, trainee_kind))
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
