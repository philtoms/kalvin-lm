"""The lean training harness.

A minimal, synchronous, non-judging loop. Compile a KScript source, feed each
compiled entry to the engine one at a time, and present the engine's
``(batch, observations)`` response. The harness is feeder, driver, and
presenter — it never judges. The trainer (a pi agent, outside the loop) reads
the trace, makes decisions, edits the engine and/or the source, and re-runs.

Usage::

    PYTHONPATH=src python -m dialogue.harness path/to/script.ks
    PYTHONPATH=src python -m dialogue.harness path/to/script.ks -v
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from dialogue.engine import Engine
from dialogue.engine_state import EngineState
from kalvin.kline import KLine, KNode
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.significance import SIG_MASK, SIG_S1, SIG_S3, SIG_S4, BandLayout
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from training.trainer.curriculum_document import (
    CurriculumDocument,
    CurriculumParseError,
)

_BAND_ORDER = ("S1", "S2", "S3", "S4")

# Classifies an ask's significance byte into its band on the S1–S4 spectrum.
_LAYOUT = BandLayout()


@dataclass
class Turn:
    """One engine call within a step: the feeds and its response."""

    feeds: list[KValue]
    grounds: list[KValue] = field(default_factory=list)
    asks: list[KValue] = field(default_factory=list)
    #: The supervisor's responses to this turn's escalated proposals,
    #: paired with the proposal index in ``asks`` they answer.
    escalations: list[tuple[int, KValue]] = field(default_factory=list)


@dataclass
class StepResult:
    """One loop iteration: the input entry, the engine's asks, and the
    ratifying answers the harness fed back. """

    index: int
    entry: KValue
    turns: list[Turn] = field(default_factory=list)
    answers: list[KValue] = field(default_factory=list)
    stopped_on: KValue | None = None


class Harness:
    """Compile a source, then feed each compiled entry to the engine in turn.

    The engine and its state are the only participants. The harness holds no
    verdict logic — it records what the engine returned and hands it to the
    presenter. Both ``tokenizer`` and ``engine`` are supplied fully
    constructed; see :func:`make_engine` / :func:`load_engine` for the
    single construction path.
    """

    def __init__(
        self,
        tokenizer: NLPTokenizer,
        engine: Engine,
        escalate: Callable[[KValue], KValue] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._engine = engine
        # The escalation point: an ask the script cannot answer goes to the
        # supervisor, who decides its significance (ratify at S1 to ground it;
        # decline at S4 to refuse). Default: decline.
        self._escalate = escalate or (lambda ask: KValue(ask.kline, SIG_S4))

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def state(self) -> EngineState:
        return self._engine.state

    @property
    def signifier(self) -> NLPSignifier:
        # make_engine/load_engine store an NLPSignifier; the state types it as
        # KSignifier.
        return cast(NLPSignifier, self._engine.state.signifier)

    def run(self, source: str) -> list[StepResult]:
        """Compile ``source``, open each sub-script's dialogue, and let the
        engine drive.

        Each sub-script (annotation group) is opened with its first entry.
        From there the engine asks; the harness answers each ask from the
        script or the run stops.
        """
        entries = compile_source(
            source, tokenizer=self._tokenizer, signifier=self.signifier, dev=True
        )
        tokens = {
            sig: word
            for sig, word in self._single_token_labels(source).items()
        }
        # Authored sub-scripts: scope-0 entries preserve authored order — a
        # group is a maximal run sharing an annotation ('' joins the current
        # run; a repeated annotation is a distinct authored group). Scope-1/2
        # entries (MTS/identities) are allocated to the first group with a
        # matching, not-yet-served occurrence of their annotation; '' joins
        # the preceding entry's group.
        groups: list[tuple[str, list[KValue]]] = []
        by_key: dict[str, list[KValue]] = {}
        current: list[KValue] | None = None
        for entry in entries:
            annotation = entry.kline.dbg.annotation if entry.kline.dbg else ""
            scope = entry.kline.dbg.scope if entry.kline.dbg else 0
            if scope == 0:
                if current is None or (
                    annotation and annotation != groups[-1][0].rsplit("#", 1)[0]
                ):
                    occurrence = sum(
                        1 for k, _ in groups if k.rsplit("#", 1)[0] == annotation
                    )
                    key = f"{annotation}#{occurrence}"
                    groups.append((key, []))
                    current = groups[-1][1]
                    by_key.setdefault(key, current)
                current.append(entry)
            else:
                # MTS entries dedup globally: one set per compound. The whole
                # set belongs to the first occurrence of its annotation.
                if annotation:
                    match = next(
                        (
                            f"{annotation}#{i}"
                            for i in range(
                                sum(
                                    1 for k, _ in groups
                                    if k.rsplit("#", 1)[0] == annotation
                                )
                            )
                            if f"{annotation}#{i}" in by_key
                        ),
                        None,
                    )
                    target = by_key.get(match) if match else None
                else:
                    target = None
                if target is None and annotation and not any(
                    k.rsplit("#", 1)[0] == annotation for k, _ in groups
                ):
                    # A bare annotated sig (empty sub-script body): the only
                    # entry it produced is this one — it opens its own group.
                    key = f"{annotation}#0"
                    groups.append((key, []))
                    target = groups[-1][1]
                    by_key[key] = target
                if target is None:
                    # No matching group yet (or ''): the nearest preceding
                    # scope-1/2 entry's group, else the first group.
                    target = next(
                        (
                            g for k, g in reversed(list(by_key.items()))
                            if any(
                                (e.kline.dbg.scope if e.kline.dbg else 0) != 0
                                for e in g
                            )
                        ),
                        groups[0][1] if groups else None,
                    )
                if target is not None:
                    target.append(entry)
        # The answering pools grow as groups open — the harness never answers
        # from a sub-script the dialogue has not reached (no look-ahead).
        heads: dict[int, list[KValue]] = {}
        exact: dict[tuple[int, tuple[int, ...]], KValue] = {}
        words: set[int] = set(tokens)
        # An opening group is a step; its entries join the answering pools
        # when it opens (cumulative — past and current groups only).
        steps: list[tuple[str, list[KValue], KValue]] = []
        for key, group in groups:
            # The opener is the group's question: a scope-0 authored entry,
            # else the canon itself (a bare annotated sig's MTS canon) —
            # never an identity. An identity is an answer, not a question;
            # opening with it grounds the group's words before the question
            # is ever asked.
            opener = next(
                (e for e in group if e.kline.dbg and e.kline.dbg.scope == 0),
                next(
                    (
                        e for e in group
                        if e.kline.dbg and e.kline.dbg.op == "CANONIZES"
                    ),
                    group[0],
                ),
            )
            annotation = opener.kline.dbg.annotation if opener.kline.dbg else ""
            if not annotation:
                continue
            steps.append((key, group, opener))
        results: list[StepResult] = []
        for i, (key, group, opener) in enumerate(steps):
            # Fresh answers per authored group: a repeated group is a second
            # ask, not a replay of the first one's dedup ledger.
            answered: set[tuple[int, tuple[int, ...]]] = set()
            for entry in group:
                kline = entry.kline
                heads.setdefault(kline.signature, []).append(entry)
                exact.setdefault(
                    (kline.signature, tuple(kline.nodes)), entry
                )
                observe = getattr(self._escalate, "observe", None)
                if observe is not None:
                    observe(entry)
                if (
                    kline.nodes != [kline.signature]
                    and kline.signature == self.signifier.signature_of(kline.nodes)
                ):
                    # A compound self-ref (e.g. DH unpacking to did, have):
                    # its nodes are script-known words.
                    words.update(kline.nodes)
            step = StepResult(i, opener)
            results.append(step)
            queue: list[list[KValue]] = [[opener]]
            while queue:
                feeds = queue.pop(0)
                batch, observations = self._engine.rationalise(feeds)
                deduped = _dedup(batch)
                replies: list[KValue] = []
                turn = Turn(feeds, observations, deduped)
                step.turns.append(turn)
                for ask_i, ask in enumerate(deduped):
                    if self.state.is_grounded(ask.kline):
                        # K stating knowledge it already holds — not a
                        # question. No reply, no escalation.
                        continue
                    reply = self._answer(ask, heads, exact, words, answered)
                    if reply is None:
                        if not ask.kline.nodes:
                            # An empty ask is signature discovery, not a
                            # proposal — nothing for a supervisor to decide.
                            replies.append(KValue(ask.kline, SIG_S4))
                            continue
                        # Off-script: escalate — the supervisor decides.
                        response = self._escalate(ask)
                        turn.escalations.append((ask_i, response))
                        replies.append(response)
                        continue
                    replies.extend(reply)
                if replies:
                    step.answers.extend(replies)
                    queue.append(replies)
        return results

    def _single_token_labels(self, source: str) -> dict[int, str]:
        """``{signature: word}`` for every single-token word in the source.

        A word that encodes to one typed token is its own signature, so its
        identity ``X:[X]`` is derivable from the tokenizer alone (e.g. an
        abbreviation RHS like ``M(od)`` that never heads a script entry).
        """
        from ks.compiler import Compiler
        from ks.lexer import Lexer
        from ks.parser import Parser
        compiler = Compiler(self._tokenizer, signifier=self.signifier, dev=True)
        compiler.compile(Parser(Lexer(source).tokenize()).parse())
        return {
            sig: word
            for sig, word in compiler.node_labels.items()
            if self._tokenizer.encode(word) == [sig]
        }

    def _answer(
        self,
        ask: KValue,
        heads: dict[int, list[KValue]],
        exact: dict[tuple[int, tuple[int, ...]], KValue],
        words: set[int],
        answered: set[tuple[int, tuple[int, ...]]],
    ) -> list[KValue] | None:
        """The ratifying reply to ``ask``, or ``None`` when the script cannot
        answer it.

        An identity ask ``X:[]`` is answered by an identity ``X:[X]`` plus the
        script klines headed ``X``. A proposal ask ``A:[B]`` is answered by the
        matching script kline plus its countersignature ``B:[A]``.
        """
        kline = ask.kline
        key = (kline.signature, tuple(kline.nodes))
        if key in answered:
            return []
        answered.add(key)
        if not kline.nodes:
            script_klines = [
                e for e in heads.get(kline.signature, [])
                if e.kline.nodes != [kline.signature]
                # K already holds it: grounded, or attending to it in STM
                # (re-feeding the asked question re-arms a refused ask).
                and not self.state.is_grounded(e.kline)
                and not any(
                    entry.signature == e.kline.signature
                    and entry.nodes == e.kline.nodes
                    for entry in self.state.stm
                )
            ]
            is_word = kline.signature in words or any(
                e.kline.nodes == [kline.signature]
                for e in heads.get(kline.signature, [])
            )
            if not script_klines and not is_word:
                return None
            # Only terminals (single-token words) get a generated identity;
            # a non-terminal's word form must come from the script.
            if not is_word:
                return [*script_klines]
            identity = KValue(
                KLine(kline.signature, [kline.signature]), SIG_S1
            )
            return [identity, *script_klines]
        hit = exact.get(key)
        if hit is None:
            # An identity proposal X:[X] for a terminal word is the
            # tokenizer's own fact — answer it without the supervisor.
            if (
                kline.nodes == [kline.signature]
                and kline.signature in words
            ):
                return [KValue(kline, SIG_S1)]
            return None
        countersigns = [
            KValue(KLine(node, [kline.signature]), SIG_S3)
            for node in kline.nodes
        ]
        return [hit, *countersigns]


# ── Construction (single source of truth) ────────────────────────────────
#
# The factories wire signifier → state → engine → harness so the signifier
# lives in one place (the EngineState). The engine constructs its own S2
# strategy (ExpandFit) over the state.

def make_engine(
    tokenizer: NLPTokenizer,
) -> Harness:
    """Build a harness over a fresh state: new signifier → state → engine."""
    state = EngineState(NLPSignifier())
    return Harness(tokenizer, Engine(state))


def load_engine(
    path: str | Path,
    tokenizer: NLPTokenizer,
) -> Harness:
    """Build a harness over a loaded prior state (reusing its signifier)."""
    signifier = NLPSignifier()
    state = EngineState.load(signifier, path)
    return Harness(tokenizer, Engine(state))


# ── Presentation ──────────────────────────────────────────────────────────


def _dedup(batch: list[KValue]) -> list[KValue]:
    """First occurrence of each (signature, nodes, band) in emission order."""
    seen: set[tuple[int, tuple[int, ...], int]] = set()
    out: list[KValue] = []
    for v in batch:
        key = (v.kline.signature, tuple(v.kline.nodes), v.significance)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _label(signature: int, labels: dict[int, str], verbose: bool) -> str:
    name = labels.get(signature)
    if verbose:
        return f"{name}|0x{signature:x}" if name else f"0x{signature:x}"
    return name or f"0x{signature:x}"


def _render_kline_struct(kline: KLine, labels: dict[int, str], verbose: bool) -> str:
    sig = _label(kline.signature, labels, verbose)
    nodes = ", ".join(_label(n, labels, verbose) for n in kline.nodes)
    return f"{sig}:[{nodes}]"


def _render_kline(value: KValue, labels: dict[int, str], verbose: bool) -> str:
    return _render_kline_struct(value.kline, labels, verbose)


def _band(value: KValue) -> str:
    return _LAYOUT.classify(value.significance)


def _sig_display(value: KValue) -> str:
    return f"{_band(value)} {value.significance & SIG_MASK}"


def _sig_to_label(source: str, tokenizer: NLPTokenizer, signifier: NLPSignifier) -> dict[int, str]:
    """Recompile once to recover ``{signature: scripted label}`` for display.

    Compiled-entry labels are authoritative; the encoder's ``node_labels``
    supplies words for single-token node values that never head an entry
    (e.g. MTS-expanded words like ``did``/``have``).
    """
    from ks.compiler import Compiler
    from ks.lexer import Lexer
    from ks.parser import Parser
    compiler = Compiler(tokenizer, signifier=signifier, dev=True)
    entries = compiler.compile(Parser(Lexer(source).tokenize()).parse())
    out: dict[int, str] = dict(compiler.node_labels)
    for e in entries:
        d = e.kline.dbg
        if d is None:
            continue
        label = d.label
        if label:
            out.setdefault(e.kline.signature, label)
    return out


def _render_step(step: StepResult, labels: dict[int, str], verbose: bool) -> str:
    lines = [f"── Step {step.index + 1}  in  {_band(step.entry)}  "
             f"{_render_kline(step.entry, labels, verbose)} ──"]
    for t, turn in enumerate(step.turns, 1):
        feeds = " + ".join(
            f"{_render_kline(v, labels, verbose)} {_band(v)}" for v in turn.feeds
        )
        lines.append(f"  T{t:02d}  feed    {feeds}")
        for v in turn.grounds:
            lines.append(f"        grounds {_render_kline(v, labels, verbose)}")
        escalations = {i: r for i, r in turn.escalations}
        for i, v in enumerate(turn.asks):
            if v.kline.nodes:
                lines.append(
                    f"        {'proposes':<8} {_render_kline(v, labels, verbose)} {_sig_display(v)}"
                )
            else:
                lines.append(f"        {'asks':<8} {_render_kline(v, labels, verbose)}")
            if i in escalations:
                r = escalations[i]
                verdict = (
                    "ratified" if _band(r) == "S1" else
                    "graded" if _band(r) in ("S2", "S3") else
                    "declined"
                )
                lines.append(f"        {'supervisor':<8} {verdict} ({_band(r)})")
    if step.stopped_on is not None:
        lines.append(
            f"  stop    unanswerable ask  {_render_kline(step.stopped_on, labels, verbose)}"
        )
    return "\n".join(lines)


def _render_grounded(state: EngineState, labels: dict[int, str], verbose: bool,
                        pre_grounded: set[tuple[KNode, tuple[KNode, ...]]] | None = None) -> str:
    if not state.ltm:
        return "  (grounded nothing)"
    lines = []
    reloaded: list[str] = []
    for signature in sorted(state.ltm, key=lambda s: (s.bit_length(), s)):
        owner = _label(signature, labels, verbose)
        bucket = state.ltm[signature]
        for kl in bucket:
            nodes = ", ".join(_label(n, labels, verbose) for n in kl.nodes)
            line = f"      {owner}:[{nodes}]"
            if pre_grounded is not None and (signature, tuple(kl.nodes)) in pre_grounded:
                reloaded.append(line)
            else:
                lines.append(line)
    if pre_grounded is not None:
        lines.append("    (reloaded, held before this run)")
        lines.extend(reloaded)
    return "\n".join(lines)


def _render_stm(state: EngineState, labels: dict[int, str], verbose: bool) -> str:
    if not state.stm:
        return "  (empty)"
    return "\n".join(
        f"      {_render_kline_struct(kl, labels, verbose)}" for kl in state.stm
    )


def _render_summary(results: list[StepResult], state: EngineState,
                    labels: dict[int, str], verbose: bool,
                    pre_grounded: set[tuple[KNode, tuple[KNode, ...]]] | None = None) -> str:
    bands: Counter = Counter()
    for step in results:
        for turn in step.turns:
            for v in turn.asks:
                bands[_band(v)] += 1
    band_str = "  ".join(f"{b}={bands.get(b, 0)}" for b in _BAND_ORDER) or "-"
    return (
        f"── summary ──\n"
        f"  steps: {len(results)}\n"
        f"  asks by band: {band_str}\n"
        f"  grounded:\n{_render_grounded(state, labels, verbose, pre_grounded)}\n"
        f"  stm (attending to at end of run):\n{_render_stm(state, labels, verbose)}"
    )


def present(results: list[StepResult], state: EngineState, source: str,
            tokenizer: NLPTokenizer, signifier: NLPSignifier, *, verbose: bool,
            pre_grounded: set[tuple[KNode, tuple[KNode, ...]]] | None = None) -> None:
    labels = _sig_to_label(source, tokenizer, signifier)
    last_annotation: str | None = None
    for step in results:
        annotation = step.entry.kline.dbg.annotation if step.entry.kline.dbg else ""
        if annotation and annotation != last_annotation:
            print(f"\n[{annotation}]")
            last_annotation = annotation
        print(_render_step(step, labels, verbose))
    print()
    print(_render_summary(results, state, labels, verbose, pre_grounded))


# ── CLI ───────────────────────────────────────────────────────────────────


def _queue_supervisor(
    queue: list[str], labels: dict[int, str], verbose: bool
) -> Callable[[KValue], KValue]:
    """Answer escalations from a fixed response queue; return to the
    supervisor when it runs out.

    Lets a supervisor concentrate on successive proposals as the script
    evolves: grade the expected escalations up front, run unattended. The
    first escalation beyond the queue is prompted for — control returns to
    the supervisor, whose queued grades remain spent.
    """
    return _interactive_supervisor(labels, verbose, queue)


def _interactive_supervisor(
    labels: dict[int, str], verbose: bool, queue: list[str] | None = None
) -> Callable[[KValue], KValue]:
    """Prompt on each off-script ask: the supervisor decides significance.

    With ``queue``, its grades are spent first, unattended; the first
    escalation beyond it returns control to the prompt.

    1 ratifies at S1 (K grounds the kline in LTM — the fast path next
    time); 2/3 grade and decline; 4 (or empty) declines outright.
    """
    sig_map = {"1": SIG_S1, "2": 0x80, "3": 0x40, "4": 0}
    it = iter(queue or [])

    def supervise(ask: KValue) -> KValue:
        print(f"\n  ⚠ escalated ask: {_render_kline(ask, labels, verbose)}")
        choice = next(it, None)
        if choice is not None:
            choice = choice.strip()
            if choice not in sig_map:
                choice = ""
            sig = sig_map.get(choice, 0)
            verdict = "ratified" if sig == SIG_S1 else "declined"
            print(
                f"  supervisor ({choice or '4'}): {verdict} "
                f"({_band(KValue(ask.kline, sig))})"
            )
            return KValue(ask.kline, sig)
        while True:
            try:
                choice = input(
                    "  ratify? [1=S1 ratify  2=S2  3=S3  4/enter=decline S4] > "
                ).strip()
            except EOFError:
                choice = ""  # no supervisor input left: decline
                break
            if choice in sig_map or choice == "":
                break
            print("  enter 1, 2, 3, 4, or empty")
        sig = sig_map.get(choice, 0)
        verdict = "ratified" if sig == SIG_S1 else "declined"
        print(f"  supervisor: {verdict} ({_band(KValue(ask.kline, sig))})")
        return KValue(ask.kline, sig)

    return supervise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a script (markdown plan) or KScript source through the "
             "lean engine and present the trace.",
    )
    parser.add_argument("source", help="Path to a markdown plan file or a .ks KScript file")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show hex signatures alongside scripted labels.",
    )
    parser.add_argument(
        "-s", "--supervise", nargs="?", const="interactive", default=None,
        metavar="BANDS",
        help="Supervisor for off-script asks: a comma list of bands "
             "(e.g. '1,4,1') answers escalations from a response queue, "
             "declining once exhausted; bare '-s' prompts interactively. "
             "1 ratifies at S1 (grounds in LTM), 2/3 grade and decline, "
             "4 declines.",
    )
    parser.add_argument(
        "-e", "--structural", action="store_true",
        help="Grade off-script asks with the structural supervisor: compiler "
             "evidence (canons, countersigns, denotations) decides the band, "
             "S1 only when the script's proof completes.",
    )
    parser.add_argument(
        "-t", "--training", action="store_true",
        help="User-significance teaching: supervisor S2/S3 stamps on K's own "
             "proposals are filed as patterns/pivots and replayed for "
             "matching asks.",
    )
    parser.add_argument(
        "-p", "--persist", nargs="?", const="auto", default=None, metavar="PATH",
        help="Load engine state before the run and save it after. PATH "
             "defaults to data/dialogue/{script_name}.json.",
    )
    args = parser.parse_args(argv)

    source_path = Path(args.source)
    if source_path.suffix == ".ks":
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"harness: could not read {args.source!r}: {exc}", file=sys.stderr)
            return 2
        tok = NLPTokenizer()
        state_path = (
            Path(f"data/dialogue/{source_path.stem}.json")
            if args.persist == "auto"
            else Path(args.persist) if args.persist else None
        )
        if state_path is not None and state_path.exists():
            harness = load_engine(state_path, tok)
            n = sum(len(b) for b in harness.state.ltm.values())
            print(f"── running on reloaded state: {n} grounded klines "
                  f"from {state_path} ──")
        else:
            harness = make_engine(tok)
        if args.training:
            from dialogue.engine import Engine
            Engine.TRAINING = True
        if args.structural:
            from dialogue.structural import SemanticEvidence, StructuralSupervisor

            labels = _sig_to_label(source, tok, harness.signifier)
            render = lambda v: _render_kline(v, labels, args.verbose)  # noqa: E731
            supervisor = StructuralSupervisor(
                SemanticEvidence(harness.signifier), harness.state, render
            )
            harness._escalate = supervisor
        if args.supervise:
            labels = _sig_to_label(source, tok, harness.signifier)
            harness._escalate = (
                _interactive_supervisor(labels, args.verbose)
                if args.supervise == "interactive"
                else _queue_supervisor(args.supervise.split(","), labels, args.verbose)
            )
        pre_grounded = (
            {(sig, tuple(kl.nodes))
             for sig, bucket in harness.state.ltm.items() for kl in bucket}
            if state_path is not None and state_path.exists() else None
        )
        results = harness.run(source)
        present(results, harness.state, source, tok, harness.signifier,
                verbose=args.verbose, pre_grounded=pre_grounded)
        if state_path is not None:
            harness.state.save(state_path)
        return 0

    try:
        document = CurriculumDocument.from_file(source_path)
    except (CurriculumParseError, OSError) as exc:
        print(f"harness: could not read source {args.source!r}: {exc}", file=sys.stderr)
        return 2

    tok = NLPTokenizer()
    harness = make_engine(tok)
    cumulative = ""
    for lesson in document.lessons:
        source = "\n".join(lesson.kscript)
        cumulative = f"{cumulative}\n{source}"
        print(f"\n══ lesson {lesson.label} ══")
        results = harness.run(source)
        present(results, harness.state, cumulative, tok, harness.signifier, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
