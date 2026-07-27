"""Run a dialogue end-to-end through the script-reading trainer and trainee.

Loads a dialogue script, decodes it, and drives it to completion through the
sink-shaped :class:`Runner`, which bridges the actors' emissions onto a
:class:`MessageBus` (coverage is order-agnostic on middle entries — the natural
fit for real actors). 

``--rationalise`` substitutes a :class:`RationalisingTrainee` (a real, stateful,
rationalising trainee) for the default ``ScriptTrainee``, and ``--synthesize``
substitutes a :class:`SynthesizingTrainer` (a real trainer that derives each
turn from the compiled source) for the default ``ScriptTrainer``. The two flags
are orthogonal: passing both runs the two real actors against the same
reference script.

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
    python scripts/dialogue_run.py --rationalise-trainer       # RationalisingTrainer
    python scripts/dialogue_run.py --rationalise-both          # both rationalising actors
    python scripts/dialogue_run.py --synthesize                # SynthesizingTrainer
    python scripts/dialogue_run.py --synthesize --rationalise  # SynthesizingTrainer + RationalisingTrainee
    python scripts/dialogue_run.py --divergence                # fail (exit 1) on divergence

By default divergences are accepted and the run completes (exit 0), with any
unmatched emissions/groundings reported in the trace. Pass ``--divergence`` to
fail (exit 1) on the first immediate divergence instead. 
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
    GroundingDivergence,
    ScriptTrainee,
    ScriptTrainer,
    decode,
    decode_events,
    load_script,
    load_script_file,
    run,
)
from training.dialogue.actors import (  # noqa: E402
    RationalisingTrainee,
    RationalisingTrainer,
    SynthesizingTrainer,
)
from training.dialogue.rationalise import RationaliserState  # noqa: E402

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


def _render_trace_entry(idx: int, scripted: str, record: dict | None, verbose: bool) -> str:
    """One trace line: the scripted form, with the source JSON record beneath.

    ``record`` is the raw JSON turn dict — when present it is shown verbatim so
    the script row the author wrote is visible alongside its decoded form.
    """
    line = f"  #{idx:2} {scripted}"
    if verbose and record is not None:
        line += "\n       " + json.dumps(record, sort_keys=True)
    return line


def _trace(
    events: list,
    decoded: list,
    sig_to_label: dict[int, str],
    verbose: bool
) -> str:
    """Arrival-ordered trace of emissions in scripted form.

    Events arrive in bus-delivery order. Each is rendered on its own numbered
    line via :func:`_scripted_form_event`, and — when the emission matches a
    decoded turn's content key — the script's JSON record for that turn is shown
    beneath it (the verbatim row the author wrote). An emission with no
    matching decoded turn shows no record.
    """
    # Index decoded turns by content key as ordered queues, so each
    # emission of a key associates with the next decoded turn of that key in
    # script order (the Nth emission ↔ the Nth occurrence). A close that
    # recurs as coverage is consumed in order: the 1st emission shows the
    # coverage row, the 2nd shows the close row.
    from collections import defaultdict, deque

    from training.dialogue.decoder import turn_content_key

    records_by_key: dict = defaultdict(deque)
    for turn in decoded:
        if turn.close:
            continue  # close is terminal (its own slot); exclude from queue
        records_by_key[turn_content_key(turn)].append(turn.record)

    def _key_of(ev) -> tuple:
        return (
            ev.role or "?",
            ev.proposal.kline.signature,
            tuple(ev.proposal.kline.nodes),
            ev.proposal.significance,
        )

    lines = []
    for i, ev in enumerate(events):
        key = _key_of(ev)
        queue = records_by_key.get(key)
        record = queue.popleft() if queue else None
        lines.append(
            _render_trace_entry(i, _scripted_form_event(ev, sig_to_label), record, verbose)
        )
    # Displacement: authored coverage copies never emitted. The per-key
    # queues were popped once per emission, so leftover entries are authored
    # rows never consumed. The close is terminal (its slot pops only when the
    # run closes on it), so it is never reported here as displaced.
    unreached = []
    for turn in decoded:
        if turn.close:
            continue
        q = records_by_key.get(turn_content_key(turn))
        if q:
            unreached.append(turn)
    if unreached:
        lines.append("  --- displacement (uncovered) ---")
        idx = len(events)
        for turn in unreached:
            op = turn.op if turn.op else "?"
            scripted = _render_scripted(
                turn.role, op, turn.value, sig_to_label, close=turn.close
            )
            lines.append(_render_trace_entry(idx, scripted, turn.record, verbose))
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


def _render_grounded(
    actor: RationalisingTrainee | RationalisingTrainer,
    sig_to_label: dict[int, str],
) -> str:
    """A rationalising actor's grounded klines after the run, in scripted form.

    ``RationaliserState.grounded`` is ``{signature: [KLine, ...]}`` — every
    kline the actor has grounded at S1, deduplicated by nodes. Rendered one per
    line as ``S:[nodes]`` with labels recovered from ``sig_to_label`` (falling
    back to hex), grouped under each owning signature. Identities (empty
    nodes) render as ``S:[]``.
    """
    state = actor._state  # noqa: SLF001 — post-run inspection by the script
    if not state.grounded:
        return "  (grounded nothing)"
    lines = []
    for signature in sorted(state.grounded, key=lambda s: (s.bit_length(), s)):
        owner = sig_to_label.get(signature) or f"0x{signature:x}"
        bucket = state.grounded[signature]
        if len(bucket) == 1:
            lines.append(f"  {owner}: " + _render_nodes(bucket[0], sig_to_label))
        else:
            lines.append(f"  {owner}:")
            for kl in bucket:
                lines.append("    - " + _render_nodes(kl, sig_to_label))
    return "\n".join(lines)


def _render_nodes(kline, sig_to_label: dict[int, str]) -> str:
    """Render a kline's nodes as ``[a, b, c]`` (or ``[]`` for identities)."""
    if not kline.nodes:
        return "[]"
    labels = [sig_to_label.get(n) or f"0x{n:x}" for n in kline.nodes]
    return "[" + ", ".join(labels) + "]"


def _render_grounding_divergence(
    exc: GroundingDivergence, sig_to_label: dict[int, str]
) -> str:
    """Human-readable grounding-divergence report (white-box).

    Inverts the unexpected/over-budget K grounding and the still-unconsumed
    expected groundings into scripted form, mirroring
    :func:`_render_divergence` for the grounding channel.
    """
    grounded_scripted = _render_scripted("K", "ground", exc.grounded, sig_to_label)
    if exc.reason == "exhausted":
        verdict = "which exhausts its expected grounding budget"
    elif exc.reason == "missing":
        verdict = "an asserted grounding that was never observed"
    else:
        verdict = "which matches no expected grounding"
    header = f"FAIL — K grounding divergence: {verdict}: {grounded_scripted}."
    lines = [header]
    if not exc.unconsumed:
        lines.append("  (no other asserted groundings were unobserved.)")
    else:
        lines.append("  Unobserved asserted groundings:")
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
    parser.add_argument(
        "--rationalise-trainer",
        action="store_true",
        help=(
            "Substitute a RationalisingTrainer (a real trainer that shares the "
            "rationaliser engine with RationalisingTrainee and leads the "
            "dialogue) for the default ScriptTrainer. Orthogonal to "
            "--rationalise and --synthesize."
        ),
    )
    parser.add_argument(
        "--rationalise-both",
        action="store_true",
        help=(
            "Run both sides as rationalising actors (RationalisingTrainer + "
            "RationalisingTrainee). Shorthand for --rationalise "
            "--rationalise-trainer; mutually exclusive with the other "
            "actor-substitution flags."
        ),
    )
    parser.add_argument(
        "--divergence",
        action="store_true",
        help=(
            "Fail (exit 1) on an immediate divergence. Off by default: "
            "divergences are accepted and the run completes, with displacement "
            "and any divergence reported in the trace. Maps to the runner's "
            "on_divergence='fail'; the default maps to on_divergence='accept'."
        ),
    )
    parser.add_argument(
        "--load", metavar="PATH", default=None,
        help=(
            "Inject a saved RationaliserState into every rationalising actor "
            "at construction (a grounded prior). Defaults to "
            "data/dialogue/{script-stem}.json when a rationalising actor runs "
            "and that file exists; --load '' disables the default lookup."
        ),
    )
    parser.add_argument(
        "--save", metavar="PATH", nargs="?", const="__default__", default=None,
        help=(
            "After the run, write each rationalising actor's final state "
            "(a grounded prior for a later --load). With no path, defaults "
            "to data/dialogue/{script-stem}.json. Omit the flag to skip saving."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "After the run, list the klines K grounded (only meaningful with "
            "--rationalise; the default ScriptTrainee carries no grounded state)."
        ),
    )
    args = parser.parse_args(argv)

    dialogue_path = args.dialogue
    tok = NLPTokenizer()
    sigf = NLPSignifier()

    script = load_script(json.loads(Path(dialogue_path).read_text()))
    # The run sequence: each prior is one run, then the target script's own
    # turns are the final run. Each run is decoded independently with its
    # own coverage set and close; the same rationaliser *state* persists
    # across runs (a prior run's grounded knowledge feeds the next).
    run_scripts = [load_script_file(p) for p in script.priors] + [script]
    decoded = decode(script, tokenizer=tok, signifier=sigf)
    expected_groundings = decode_events(script, tokenizer=tok, signifier=sigf)
    # The compiled source's signature→label map (target script; shared source
    # in the canonical case, so labels resolve for every run).
    sig_to_label = _sig_to_label(script, tokenizer=tok, signifier=sigf)

    # --divergence opts into fail-on-divergence; the default accepts
    # divergences so the run completes and displacement is reported.
    on_divergence = "fail" if args.divergence else "accept"

    # The --synthesize/--rationalise/--rationalise-trainer/--rationalise-both
    # flags substitute real actors. The individual flags are orthogonal
    # (each selects one side as a real actor against a table counterpart);
    # --rationalise-both selects both rationalising actors at once and is
    # mutually exclusive with the others. The real actors are drop-in.
    if args.rationalise_both and (args.rationalise or args.rationalise_trainer
                                  or args.synthesize):
        parser.error(
            "--rationalise-both is mutually exclusive with --rationalise, "
            "--rationalise-trainer, and --synthesize (it selects both actors)"
        )
    if args.synthesize and args.rationalise_trainer:
        parser.error("--synthesize and --rationalise-trainer are mutually exclusive "
                     "(both substitute the trainer)")
    # Resolve which sides run a real actor. --rationalise-both turns on both
    # rationalising actors; otherwise each flag selects its own side.
    rationalise_trainee = args.rationalise or args.rationalise_both
    rationalise_trainer = args.rationalise_trainer or args.rationalise_both

    # Build the compiled source once for the target script when a real trainer
    # needs it. (Each run recompiles from its own source in the sequencer.)
    compiled = None
    if args.synthesize or rationalise_trainer:
        compiled = compile_source(script.source, tokenizer=tok, signifier=sigf, dev=True)

    # --load: inject a saved RationaliserState (a grounded prior) into every
    # rationalising actor. Default path is data/dialogue/{stem}.json when a
    # rationalising actor runs and that file exists; ``--load ''`` disables
    # the default lookup.
    stem = Path(dialogue_path).stem
    default_state_path = Path("data/dialogue") / f"{stem}.json"
    load_path = None
    if args.load is not None:
        load_path = Path(args.load) if args.load else None
    elif (rationalise_trainee or rationalise_trainer) and default_state_path.exists():
        load_path = default_state_path
    prior_state = RationaliserState.load(load_path) if load_path else None
    if prior_state is not None:
        print(f"  loaded grounded prior : {load_path}")

    # The actor factories are per-run (each run has its own decoded table and
    # compiled source), but the rationaliser *state* is shared across the
    # sequence — a prior run's grounded knowledge feeds the next. When no
    # --load snapshot was supplied, each role still gets one fresh
    # RationaliserState carried across every run (the spec's "same instances
    # persist"); only an explicit --load seeds it from disk. Table actors
    # carry no state, so they are fresh per run.
    shared_t_state = prior_state if prior_state is not None else RationaliserState()
    shared_k_state = prior_state if prior_state is not None else RationaliserState()

    def _trainer_factory(run_decoded, run_compiled):
        if args.synthesize:
            return lambda sink: SynthesizingTrainer(
                run_compiled, sigf, sink=sink, table=run_decoded
            )
        if rationalise_trainer:
            return lambda sink: RationalisingTrainer(
                sigf, sink=sink, compiled=run_compiled,
                table=run_decoded, state=shared_t_state,
            )
        return lambda sink: ScriptTrainer(run_decoded, sink=sink)

    def _trainee_factory(run_decoded):
        if rationalise_trainee:
            return lambda sink: RationalisingTrainee(
                sigf, sink=sink, state=shared_k_state,
            )
        return lambda sink: ScriptTrainee(run_decoded, sink=sink)

    # Drive the run sequence. Each run is decoded independently; results are
    # aggregated for the trace and summary. The final run's expected groundings
    # are verified (grounding assertions attach to their owning script).
    per_run = []  # list of (script, decoded, expected, runner, res)
    try:
        for idx, run_script in enumerate(run_scripts):
            is_final = idx == len(run_scripts) - 1
            run_decoded = decode(run_script, tokenizer=tok, signifier=sigf)
            run_expected = (
                decode_events(run_script, tokenizer=tok, signifier=sigf)
                if is_final else ()
            )
            run_compiled = None
            if args.synthesize or rationalise_trainer:
                run_compiled = compile_source(
                    run_script.source, tokenizer=tok, signifier=sigf, dev=True
                )
            runner = run(
                run_decoded,
                _trainer_factory(run_decoded, run_compiled),
                _trainee_factory(run_decoded),
                expected_groundings=run_expected,
                on_divergence=on_divergence,
            )
            res = runner.run()
            per_run.append((run_script, run_decoded, run_expected, runner, res))
    except Divergence as exc:
        print(_render_divergence(exc, sig_to_label), file=sys.stderr)
        return 1
    except GroundingDivergence as exc:
        print(_render_grounding_divergence(exc, sig_to_label), file=sys.stderr)
        return 1

    # The final run's runner/actors are the ones to inspect post-sequence.
    runner = per_run[-1][3]
    res = per_run[-1][4]

    # Render each run's trace. A single-run script (no priors) prints as
    # before; a multi-run sequence labels each run.
    for idx, (_rs, rd, _re, _rr, rres) in enumerate(per_run):
        header = "Exchange (arrival order):"
        if len(per_run) > 1:
            header = f"Run {idx + 1}/{len(per_run)} — {_rs.source.splitlines()[0] if _rs.source else ''}"
        print(header + "\n" + _trace(rres.events, rd, sig_to_label, args.verbose))
    # The supervisor-load baseline: how often the rationalising trainer's
    # cogitation PASSed and it asked the supervisor. Surfaced on every run
    # (not just -v) so the trajectory is visible as the rationaliser takes on
    # more of the load — the count is the baseline for that work.
    supervisor_line = ""
    if isinstance(runner.trainer, RationalisingTrainer):
        asks, emissions = runner.trainer.supervisor_escalations()
        supervisor_line = (
            f"\n  supervisor escalations : {asks} asks, "
            f"{len(emissions)} emitted"
        )
    # Aggregate stats across the run sequence.
    total_events = sum(len(r.events) for _, _, _, _, r in per_run)
    total_unmatched = sum(len(r.unmatched) for _, _, _, _, r in per_run)
    total_uncovered = sum(len(r.uncovered) for _, _, _, _, r in per_run)
    total_uncovered_groundings = sum(
        len(r.uncovered_groundings) for _, _, _, _, r in per_run
    )
    print(
        f"\nDialogue session: {dialogue_path}"
        + (f" ({len(per_run)} runs)" if len(per_run) > 1 else "")
        + f"\n"
        f"  events received        : {total_events}\n"
        f"  unmatched emissions    : {total_unmatched}\n"
        f"  uncovered (displacement): {total_uncovered}\n"
        f"  uncovered groundings   : {total_uncovered_groundings}"
        + supervisor_line
    )
    all_unmatched = [ev for _, _, _, _, r in per_run for ev in r.unmatched]
    if all_unmatched:
        print("\n  --- unmatched emissions (accepted divergence) ---")
        for ev in all_unmatched:
            print(
                "    "
                + _scripted_form_event(ev, sig_to_label)
            )
    all_unmatched_groundings = [
        gv for _, _, _, _, r in per_run for gv in r.unmatched_groundings
    ]
    if all_unmatched_groundings:
        print("\n  --- unmatched groundings (accepted divergence) ---")
        for gv in all_unmatched_groundings:
            print("    " + _render_scripted("K", "ground", gv, sig_to_label))
    if args.verbose:
        # -v lists the grounded klines of every rationalising actor in play.
        # Under --rationalise-both both sides are rationalising and both are
        # listed. The table actors carry no grounded state.
        actors = []
        if isinstance(runner.trainer, (RationalisingTrainee, RationalisingTrainer)):
            actors.append(("T", runner.trainer))
        if isinstance(runner.trainee, (RationalisingTrainee, RationalisingTrainer)):
            actors.append(("K", runner.trainee))
        if actors:
            for label, actor in actors:
                print(f"\n{label} grounded klines:")
                print(_render_grounded(actor, sig_to_label))
        else:
            print(
                "\n(verbose: -v lists a rationalising actor's grounded klines, "
                "but neither side is a Rationalising actor — pass "
                "--rationalise, --rationalise-trainer, or --rationalise-both.)"
            )
        # When the trainer is a RationalisingTrainer, -v also lists which
        # proposals the supervisor supplied — the specific load to target next.
        if isinstance(runner.trainer, RationalisingTrainer):
            _asks, emissions = runner.trainer.supervisor_escalations()
            if emissions:
                print("\nT supervisor-supplied proposals:")
                for v in emissions:
                    print("    " + _render_scripted("T", "sup", v, sig_to_label))
            else:
                print("\n(supervisor supplied nothing — full rationaliser load)")
    # --save: write each rationalising actor's final state as a grounded
    # prior for a later --load. Only when --save was passed. With no path,
    # defaults to data/dialogue/{stem}.json.
    if args.save is not None:
        save_path = (
            Path(args.save) if args.save != "__default__"
            else default_state_path
        )
        savers = []
        if isinstance(runner.trainer, (RationalisingTrainee, RationalisingTrainer)):
            savers.append(("T", runner.trainer))
        if isinstance(runner.trainee, (RationalisingTrainee, RationalisingTrainer)):
            savers.append(("K", runner.trainee))
        if len(savers) == 1:
            savers[0][1]._state.save(save_path)  # noqa: SLF001
            print(f"\n  saved grounded prior : {save_path} ({savers[0][0]})")
        else:
            # Both sides rationalising: disambiguate by role suffix.
            for label, actor in savers:
                p = save_path.with_name(f"{save_path.stem}.{label.lower()}.json")
                actor._state.save(p)  # noqa: SLF001
                print(f"\n  saved grounded prior : {p} ({label})")
    # Reaching here means no immediate divergence was raised. 
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
