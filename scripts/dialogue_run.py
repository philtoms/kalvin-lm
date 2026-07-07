"""Run a dialogue end-to-end through the table-reading trainer and trainee.

Loads a dialogue table, decodes it, and drives it to completion. The **run
regime defaults to peer mode**: the sink-shaped :class:`PeerRunner` drives a
fresh :class:`TableTrainer` and :class:`TableTrainee`, bridging their
emissions onto it (order-agnostic on middle entries — the natural fit for
real actors). Pass ``--ordered`` to drive the bus-agnostic ordered runner
(:func:`training.dialogue.run`) instead, which enforces strict turn order.
Selecting the regime on the command line keeps the table under test untouched
— the same JSON can be run either way. Prints a PASS/FAIL summary; with
``--verbose`` it traces the exchange, showing
each validated entry in its scripted (declarative table-row) form.

For an ordered run, ``--rationalise`` substitutes a
:class:`RationalisingTrainee` (a real, stateful, rationalising trainee) for the
default ``TableTrainee`, and ``--synthesize`` substitutes a
:class:`SynthesizingTrainer` (a real trainer that derives each turn from the
compiled script) for the default ``TableTrainer``. The two flags are
orthogonal: passing both runs the two real actors against the same golden
master. They apply in both regimes — the real actors are drop-in (adapter-
driven), so peer mode with ``--rationalise`` / ``--synthesize`` runs the real
actors through the bus-subscriber relay. Note real actors do not reproduce the
table exactly, so peer mode with them typically needs ``--on-divergence
accept`` (else off-table emissions raise ``PeerDivergence``).

The runner is bus-agnostic: there is no harness message bus and no adapter
here. Bus integration arrives with the real actors.

Usage::

    python scripts/dialogue_run.py                             # default dialogue (peer mode)
    python scripts/dialogue_run.py scripts/dialogue-mhall.json # explicit path
    python scripts/dialogue_run.py --verbose                   # scripted-form trace
    python scripts/dialogue_run.py --rationalise               # RationalisingTrainee
    python scripts/dialogue_run.py --synthesize                # SynthesizingTrainer
    python scripts/dialogue_run.py --synthesize --rationalise  # both real actors
    python scripts/dialogue_run.py --ordered                   # ordered mode (strict turn order)
    python scripts/dialogue_run.py --on-divergence accept      # accept off-table emissions (peer)

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


def _sig_to_label(table, *, tokenizer, signifier) -> dict[int, str]:
    """Reverse the compiled script into a ``{signature: label}`` map.

    Used to recover the **scripted** (symbolic) form of a validated entry: a
    decoded turn carries a resolved :class:`KLine` (signature + node
    signatures as uint64s), and this map turns those back into the labels a
    table author wrote (e.g. ``MHALL``, ``Mary``, ``had``). The first compiled
    entry per signature wins, matching the decoder's own label indexing.
    """
    entries = compile_source(table.script, tokenizer=tokenizer, signifier=signifier, dev=True)
    out: dict[int, str] = {}
    for e in entries:
        d = e.kline.dbg
        if d is None:
            continue
        label = d.label or d.decoded
        if label:
            out.setdefault(e.kline.signature, label)
    return out


def _render_scripted(
    role: str,
    op: str,
    kvalue,
    sig_to_label: dict[int, str],
    *,
    close: bool = False,
) -> str:
    """The scripted (declarative table-row) version of a validated entry.

    A validated entry resolves every symbol to a :class:`KLine` (on its
    :class:`~kalvin.kvalue.KValue`); this inverts that for display,
    reconstructing the ``(role, op, signature, nodes, significance)`` row a
    table author would write, with symbolic labels recovered from
    ``sig_to_label``. A signature/node with no known label falls back to its
    hex form rather than disappearing silently. A script ``close`` marker is
    surfaced when present.

    Shared by the ordered trace (decoded turn) and the peer trace (event):
    both carry a role and a KValue; ``op`` is the dialogue vocabulary on a
    decoded turn and the proposal's structural ``dbg.op`` (or ``?``) on a peer
    event.
    """
    sig_label = sig_to_label.get(kvalue.kline.signature) or f"0x{kvalue.kline.signature:x}"
    node_labels = [
        sig_to_label.get(n) or f"0x{n:x}" for n in kvalue.kline.nodes
    ]
    nodes = "(" + ", ".join(node_labels) + ")" if node_labels else "()"
    band = _SIG_TO_BAND.get(kvalue.significance, "?")
    close_marker = " close" if close else ""
    return f"{role} {op:13} {sig_label}{nodes} {band}{close_marker}"


def _scripted_form(turn, sig_to_label: dict[int, str]) -> str:
    """Scripted form of a decoded (ordered-regime) turn."""
    return _render_scripted(turn.role, turn.op, turn.value, sig_to_label, close=turn.close)


def _scripted_form_event(event, sig_to_label: dict[int, str]) -> str:
    """Scripted form of a peer-regime emission (arrival-ordered).

    A peer event carries the emitter's ``role`` and its ``proposal`` KValue but
    not the table's ``op``; the proposal's ``dbg.op`` (the structural state the
    kline was built with) stands in, falling back to ``?`` when absent.
    """
    op = event.proposal.kline.dbg.op if event.proposal.kline.dbg else "?"
    return _render_scripted(event.role or "?", op, event.proposal, sig_to_label)


def _render_trace_entry(idx: int, scripted: str, record: dict | None) -> str:
    """One trace line: the scripted form, with the source JSON record beneath.

    Shared by the ordered and peer traces. ``record`` is the raw JSON turn dict
    carried on the decoded turn (``None`` for synthetic turns / unmatched
    events) — when present it is shown verbatim so the table row the author
    wrote is visible alongside its decoded form.
    """
    line = f"  #{idx:2} {scripted}"
    if record is not None:
        line += "\n       " + json.dumps(record, sort_keys=True)
    return line


def _trace(events: list, decoded, sig_to_label: dict[int, str]) -> str:
    """A per-turn trace showing the scripted form of each validated entry.

    Each emitted event is aligned to its decoded turn (by KLine identity) and
    rendered as the scripted row that produced it (see :func:`_scripted_form`),
    with the source JSON record shown beneath. When the run is incomplete —
    fewer events than decoded turns — the unreached table rows are appended
    in a clearly marked section so the whole table is visible.
    """
    # Reconstruct the decoded turn per emitted event from the decoded order.
    turns: list = []
    di = 0
    for ev in events:
        while di < len(decoded) and decoded[di].value.kline != ev.proposal.kline:
            di += 1
        if di < len(decoded):
            turns.append(decoded[di])
            di += 1
        else:  # pragma: no cover - defensive
            turns.append(None)

    lines = []
    for i, turn in enumerate(turns):
        if turn is None:
            lines.append(f"  #{i:2} (unmatched event)")
        else:
            lines.append(
                _render_trace_entry(i, _scripted_form(turn, sig_to_label), turn.record)
            )
    # Unreached rows: the run stopped before the table was exhausted (a real
    # actor diverged or stalled). Surface them so the full table is visible.
    unreached = decoded[di:]
    if unreached:
        lines.append("  --- unreached (run incomplete) ---")
        idx = len(turns)
        for turn in unreached:
            lines.append(
                _render_trace_entry(idx, _scripted_form(turn, sig_to_label), turn.record)
            )
            idx += 1
    return "\n".join(lines)


def _peer_trace(
    events: list,
    decoded: list,
    sig_to_label: dict[int, str],
) -> str:
    """Arrival-ordered trace of peer emissions in scripted form.

    Peer events arrive in bus-delivery order (no synchronised alternation). Each
    is rendered on its own numbered line via :func:`_scripted_form_event`, and —
    when the emission matches a decoded turn's content key — the table's JSON
    record for that turn is shown beneath it (the verbatim row the author
    wrote). An emission with no matching decoded turn (e.g. an off-table
    emission under ``on_divergence="accept"``) shows no record.
    """
    # Index decoded turns by content key so each event can be associated with
    # its source table row (peer content identity is role + kline +
    # significance — see ``turn_content_key``). The first decoded turn per key
    # wins; duplicate content collapses to one record, matching the peer
    # runner's own coverage bookkeeping.
    from training.dialogue.decoder import turn_content_key

    record_by_key: dict = {}
    for turn in decoded:
        record_by_key.setdefault(turn_content_key(turn), turn.record)

    def _key_of(ev) -> tuple:
        return (
            ev.role or "?",
            ev.proposal.kline.signature,
            tuple(ev.proposal.kline.nodes),
            ev.proposal.significance,
        )

    emitted_keys = set()
    lines = []
    for i, ev in enumerate(events):
        emitted_keys.add(_key_of(ev))
        record = record_by_key.get(_key_of(ev))
        lines.append(
            _render_trace_entry(i, _scripted_form_event(ev, sig_to_label), record)
        )
    # Unreached rows: table content whose key never appeared in an emission
    # (the run stalled or a real actor never produced it). Surface them so the
    # full table is visible. The opening and closing are positional in peer
    # mode (consumed/terminal, not coverage), so they are never "unreached".
    unreached = [
        t for t in decoded[1:-1]
        if turn_content_key(t) not in emitted_keys
        and turn_content_key(t) != turn_content_key(decoded[0])
    ]
    if unreached:
        lines.append("  --- unreached (run incomplete) ---")
        idx = len(events)
        for turn in unreached:
            lines.append(
                _render_trace_entry(idx, _scripted_form(turn, sig_to_label), turn.record)
            )
            idx += 1
    return "\n".join(lines)


def _summary(
    result,
    decoded,
    dialogue_path: str,
    verbose: bool,
    trainer_kind: str,
    trainee_kind: str,
    sig_to_label: dict[int, str],
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
        return header + "\nExchange:\n" + _trace(result.events, decoded, sig_to_label)
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
        help="Print each validated entry in its scripted (table-row) form.",
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
    parser.add_argument(
        "--ordered",
        action="store_true",
        help=(
            "Run in ordered mode: drive the bus-agnostic ordered runner, which "
            "enforces strict turn order against the table cursor. The default "
            "is peer mode (the PeerRunner, order-agnostic on middle entries)."
        ),
    )
    parser.add_argument(
        "--on-divergence",
        choices=("fail", "accept"),
        default=None,
        help=(
            "Peer-mode divergence policy (default: fail, or the table's "
            "peer.on_divergence when it declares one). Ignored in ordered mode."
        ),
    )
    args = parser.parse_args(argv)

    dialogue_path = args.dialogue
    tok = NLPTokenizer()
    sigf = NLPSignifier()

    table = load_table(json.loads(Path(dialogue_path).read_text()))
    decoded = decode(table, tokenizer=tok, signifier=sigf)
    # The scripted-form trace needs the compiled script's signature→label map;
    # build it once when --verbose is set (shared by both regimes).
    sig_to_label = (
        _sig_to_label(table, tokenizer=tok, signifier=sigf) if args.verbose else {}
    )

    # The run regime defaults to peer mode (the PeerRunner, order-agnostic
    # on middle entries — the natural fit for real actors). ``--ordered``
    # opts into the bus-agnostic ordered runner, which enforces strict turn
    # order. The regime is selected on the command line, not by the table, so
    # the JSON under test stays untouched. The divergence policy is
    # ``--on-divergence`` when given, else the table's declared
    # ``peer.on_divergence``, else ``fail``.
    if not args.ordered:
        on_divergence = args.on_divergence
        if on_divergence is None:
            on_divergence = table.peer.on_divergence if table.is_peer else "fail"
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
                on_divergence=on_divergence,
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
            f"  on divergence       : {on_divergence}\n"
            f"  events received     : {len(res.events)}\n"
            f"  complete            : {res.complete}\n"
            f"  covered (middle)    : {res.covered}\n"
            f"  unmatched emissions : {len(res.unmatched)}\n"
            f"  uncovered rows      : {len(res.uncovered)}"
        )
        if args.verbose:
            print(
                "\nExchange (arrival order):\n"
                + _peer_trace(res.events, decoded, sig_to_label)
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

    print(
        _summary(
            result,
            decoded,
            dialogue_path,
            args.verbose,
            trainer_kind,
            trainee_kind,
            sig_to_label,
        )
    )
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
