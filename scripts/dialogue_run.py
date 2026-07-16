"""Run a dialogue end-to-end through the script-reading trainer and trainee.

Loads a dialogue script, decodes it, and drives it to completion through the
sink-shaped :class:`Runner`, which bridges the actors' emissions onto a
:class:`MessageBus` (coverage is order-agnostic on middle entries — the natural
fit for real actors). 

``--rationalise`` substitutes a :class:`RationalisingTrainee` (a real, stateful,
rationalising trainee) for the default ``ScriptTrainee``, and ``--synthesize``
substitutes a :class:`SynthesizingTrainer` (a real trainer that derives each
turn from the compiled source) for the default ``ScriptTrainer``. The two flags
are orthogonal: passing both runs the two real actors against the same golden
master. 

The runner is bus-driven: it owns a :class:`MessageBus`, builds a bus-wired
:class:`EventSink` per actor, and runs the bus until a terminal condition:
the close content is seen, the coverage set is exhausted, or both actors pass
in turn. The runner tracks coverage and immediate divergence, and reports the 
**displacement** (uncovered coverage rows). 
Every ``accept`` yields at least one proposal (``burst >=1``): an actor with 
nothing substantive to say publishes a PASS, which the runner intercepts before 
matching.

Usage::

    python scripts/dialogue_run.py                             # default dialogue
    python scripts/dialogue_run.py scripts/dialogue-mhall.json # explicit path
    python scripts/dialogue_run.py --rationalise               # RationalisingTrainee
    python scripts/dialogue_run.py --synthesize                # SynthesizingTrainer
    python scripts/dialogue_run.py --synthesize --rationalise  # both real actors

Exit code is 1 only on an immediate divergence; 0 otherwise. 
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
    Divergence,
    ScriptTrainee,
    ScriptTrainer,
    decode,
    load_script,
    run,
)
from training.dialogue.decoder import primaries_from_source  # noqa: E402
from training.dialogue.actors import (  # noqa: E402
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


def _sig_to_label(script, *, tokenizer, signifier) -> dict[int, str]:
    """Reverse the compiled source into a ``{signature: label}`` map.

    Used to recover the **scripted** (symbolic) form of a validated entry: a
    decoded turn carries a resolved :class:`KLine` (signature + node
    signatures as uint64s), and this map turns those back into the labels a
    script author wrote (e.g. ``MHALL``, ``Mary``, ``had``). The first compiled
    entry per signature wins, matching the decoder's own label indexing.
    """
    entries = compile_source(script.source, tokenizer=tokenizer, signifier=signifier, dev=True)
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
    """The scripted (declarative script-row) version of a validated entry.

    A validated entry resolves every symbol to a :class:`KLine` (on its
    :class:`~kalvin.kvalue.KValue`); this inverts that for display,
    reconstructing the ``(role, op, signature, nodes, significance)`` row a
    script author would write, with symbolic labels recovered from
    ``sig_to_label``. A signature/node with no known label falls back to its
    hex form rather than disappearing silently. A script ``close`` marker is
    surfaced when present.
    """
    sig_label = sig_to_label.get(kvalue.kline.signature) or f"0x{kvalue.kline.signature:x}"
    node_labels = [
        sig_to_label.get(n) or f"0x{n:x}" for n in kvalue.kline.nodes
    ]
    nodes = "[" + ", ".join(node_labels) + "]" if node_labels else "[]"
    band = _SIG_TO_BAND.get(kvalue.significance, "?")
    close_marker = " close" if close else ""
    return f"{role} {op:13} {sig_label}:{nodes} {band}{close_marker}"


def _scripted_form_event(event, sig_to_label: dict[int, str]) -> str:
    """Scripted form of an emission (arrival-ordered).

    An event carries the emitter's ``role`` and its ``proposal`` KValue but
    not the script's ``op``; the proposal's ``dbg.op`` (the structural state the
    kline was built with) stands in, falling back to ``?`` when absent. A PASS
    (the no-content proposal, DDT-22) renders as ``<role> PASS``.
    """
    from training.dialogue.runner import is_pass

    if is_pass(event):
        return f"{event.role or '?'} PASS"
    op = event.proposal.kline.dbg.op if event.proposal.kline.dbg else "?"
    return _render_scripted(event.role or "?", op, event.proposal, sig_to_label)


def _render_trace_entry(idx: int, scripted: str, record: dict | None) -> str:
    """One trace line: the scripted form, with the source JSON record beneath.

    ``record`` is the raw JSON turn dict — when present it is shown verbatim so
    the script row the author wrote is visible alongside its decoded form.
    """
    line = f"  #{idx:2} {scripted}"
    if record is not None:
        line += "\n       " + json.dumps(record, sort_keys=True)
    return line


def _trace(
    events: list,
    decoded: list,
    sig_to_label: dict[int, str],
) -> str:
    """Arrival-ordered trace of emissions in scripted form.

    Events arrive in bus-delivery order. Each is rendered on its own numbered
    line via :func:`_scripted_form_event`, and — when the emission matches a
    decoded turn's content key — the script's JSON record for that turn is shown
    beneath it (the verbatim row the author wrote). An emission with no
    matching decoded turn shows no record.
    """
    # Index decoded turns by content key so each event can be associated with
    # its source script row (content identity is role + kline + significance —
    # see ``turn_content_key``). The first decoded turn per key wins; duplicate
    # content collapses to one record, matching the runner's own coverage
    # bookkeeping.
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
    # Displacement: coverage rows whose key never appeared in an emission
    # (the runner's ``uncovered`` — how far the realized dialogue fell short
    # of the authored whole-exchange coverage). The close is the ``close:true``
    # turn or the last row; it is terminal, not coverage, so it is never
    # "displaced".
    close_idx = next((i for i, t in enumerate(decoded) if t.close), len(decoded) - 1)
    unreached = [
        t for i, t in enumerate(decoded)
        if i != close_idx and turn_content_key(t) not in emitted_keys
    ]
    if unreached:
        lines.append("  --- displacement (uncovered) ---")
        idx = len(events)
        for turn in unreached:
            op = turn.op if turn.op else "?"
            scripted = _render_scripted(
                turn.role, op, turn.value, sig_to_label, close=turn.close
            )
            lines.append(_render_trace_entry(idx, scripted, turn.record))
            idx += 1
    return "\n".join(lines)


def _render_divergence(exc: Divergence, sig_to_label: dict[int, str]) -> str:
    """Human-readable divergence report.

    The raw :class:`Divergence` carries signatures as hex and a bare count of
    uncovered rows — opaque without the compiled sources's label map. This
    inverts the diverging emission, the last healthy coverage match, and the
    still-uncovered same-role rows into their scripted (script-row) form, so an
    author can read what the actor said versus what it was still expected to
    say. Two divergence reasons are distinguished: ``exhausted`` (the content
    is in the script but every authored copy was already consumed — duplicate-
    key exhaustion) and ``unmatched`` (the content matches no row). An
    ``unconsumed`` entry built by the runner from a content key has no
    ``record`` (it is a placeholder), so only its scripted form is shown.
    """
    emitted_scripted = _render_scripted(exc.role, "emit", exc.emitted, sig_to_label)
    if exc.reason == "exhausted":
        verdict = "which exhausts its authored coverage budget"
    else:
        verdict = "which matches no closing or middle row"
    header = f"FAIL — {exc.role} divergence: emitted {emitted_scripted}, {verdict}."
    lines = [header]
    if exc.last_coverage_event is not None:
        last = exc.last_coverage_event
        last_role = last.role or "?"
        last_scripted = _render_scripted(
            last_role, "cover", last.proposal, sig_to_label
        )
        lines.append(f"  Last healthy coverage match: {last_scripted}")
    if not exc.unconsumed:
        lines.append("  (no same-role rows remained uncovered.)")
    else:
        lines.append(f"  Still-uncovered {exc.role} rows at divergence:")
        for turn in exc.unconsumed:
            op = turn.op if turn.op else "?"
            lines.append(
                "    " + _render_scripted(turn.role, op, turn.value, sig_to_label)
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a dialogue end-to-end through the script-reading actors."
    )
    parser.add_argument(
        "dialogue",
        nargs="?",
        default=DEFAULT_DIALOGUE,
        help=f"Path to a dialogue JSON (default: {DEFAULT_DIALOGUE})",
    )
    parser.add_argument(
        "--rationalise",
        action="store_true",
        help=(
            "Substitute a RationalisingTrainee (a real, stateful rationalising trainee) "
            "for the default ScriptTrainee. The trainer stays a ScriptTrainer "
            "(the deterministic oracle)."
        ),
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help=(
            "Substitute a SynthesizingTrainer (a real trainer that derives each "
            "turn from the compiled source) for the default ScriptTrainer. "
            "Orthogonal to --rationalise."
        ),
    )
    args = parser.parse_args(argv)

    dialogue_path = args.dialogue
    tok = NLPTokenizer()
    sigf = NLPSignifier()

    script = load_script(json.loads(Path(dialogue_path).read_text()))
    decoded = decode(script, tokenizer=tok, signifier=sigf)
    # The compiled source's signature→label map.
    sig_to_label = _sig_to_label(script, tokenizer=tok, signifier=sigf)

    # Always on
    on_divergence = "fail"

    # The --synthesize/--rationalise flags substitute real actors. They are
    # orthogonal: passing both runs the two real actors against the same golden
    # master. The real actors are drop-in (adapter-driven).
    compiled = None
    primaries = None
    if args.synthesize:
        compiled = compile_source(script.source, tokenizer=tok, signifier=sigf, dev=True)
        primaries = primaries_from_source(
            script.source, tokenizer=tok, signifier=sigf
        )

    trainer_factory = (
        (lambda sink: SynthesizingTrainer(compiled, sigf, primaries, sink=sink))
        if args.synthesize
        else (lambda sink: ScriptTrainer(decoded, sink=sink))
    )
    trainee_factory = (
        (lambda sink: RationalisingTrainee(sigf, sink=sink))
        if args.rationalise
        else (lambda sink: ScriptTrainee(decoded, sink=sink))
    )
    try:
        runner = run(
            decoded,
            trainer_factory,
            trainee_factory,
            on_divergence=on_divergence,
        )
        res = runner.run()
    except Divergence as exc:
        print(_render_divergence(exc, sig_to_label), file=sys.stderr)
        return 1

    print(
        "Exchange (arrival order):\n"
        + _trace(res.events, decoded, sig_to_label)
    )
    print(
        f"\nDialogue session: {dialogue_path}\n"
        f"  events received     : {len(res.events)}\n"
        # f"  unmatched emissions : {len(res.unmatched)}\n"
        f"  uncovered (displacement): {len(res.uncovered)}"
    )
    # Reaching here means no immediate divergence was raised. 
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
