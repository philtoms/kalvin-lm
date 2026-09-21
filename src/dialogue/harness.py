"""The dialogue training harness.

A minimal, synchronous, non-judging loop. Compile a KScript source, feed
compiled entry to the engine one block at a time, and present the engine's
response. The harness is feeder, driver, and
presenter — it never judges. The trainer (a pi agent, outside the loop) reads
the trace, makes decisions, edits the engine and/or the source, and re-runs.

Usage::

    PYTHONPATH=src python -m dialogue.harness path/to/script.ks
    PYTHONPATH=src python -m dialogue.harness path/to/script.ks -v
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from dialogue.engine import Engine
from dialogue.engine_state import EngineState
from kalvin.kline import (
    KLine,
    KNode,
    canon_key,
    classify_misfit,
    is_ask,
    is_canon,
    is_connotation,
    is_denotation,
    is_identity,
    is_relationship,
    is_unknown,
)
from kalvin.kvalue import KValue
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.significance import (
    SIG_MASK,
    SIG_S1,
    SIG_S3,
    SIG_S4,
    BandLayout,
    gamma_to_byte,
    word_atom_count,
)
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
        tokenizer: BPETokenizer,
        engine: Engine,
        escalate: Callable[[KValue], KValue] | None = None,
        scaffolding: Literal["batch", "on-demand"] = "batch",
    ) -> None:
        self._tokenizer = tokenizer
        self._engine = engine
        # How a group's scaffolding reaches the engine: "batch" delivers
        # all compiled klines with the opening entry (current); "on-demand"
        # feeds only the opener and releases scaffolding as K asks for it.
        self.scaffolding = scaffolding
        # The escalation point: an ask the script cannot answer goes to the
        # supervisor, who decides its significance (ratify at S1 to ground it;
        # decline at S4 to refuse). Default: decline.
        self._escalate = escalate or (lambda ask: KValue(ask.kline, SIG_S4))
        # The word→bit table compiles share; persisted with the state so a
        # reloaded state's node values mean the same words.
        self.word_bits: dict[str, int] = {}
        # The prior state's words in acquisition order — the compile's
        # outermost word list, binding chars the script cannot bind itself.
        self.known_words: list[str] = []

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

        Each sub-script (annotation group) opens with its opener. In
        ``"batch"`` scaffolding the group's remaining entries are fed first,
        priming K, and the opener follows once their asks have settled; in
        ``"on-demand"`` only the opener is fed and the group's remaining
        entries answer the engine's asks. From there the engine asks; the
        harness answers each ask from the script or the run stops.
        """
        entries = compile_source(
            source, tokenizer=self._tokenizer, signifier=self.signifier, dev=True,
            word_bits=self.word_bits, known_words=self.known_words,
        )
        # A `==` ask feeds at its subjective significance toward the goal
        # (Def 20): fresh content carries zero depths, so γ(A, B) = J of the
        # two contents. The structural band (the ask's S4 shape) is
        # recomputed from structure by the engine — never the fed byte.
        # The ask signature carries the ASK marker; goals key on the marked
        # out base so the ask and its canon look up the same goal.
        goals: dict[int, KValue] = {}
        for e in entries:
            d = e.kline.dbg
            if d is None or d.op != "ASK" or not d.goal:
                continue
            goal = next(
                (
                    g for g in entries
                    if g.kline.dbg and g.kline.dbg.label == d.goal
                    and g.kline.nodes
                ),
                None,
            )
            if goal is not None:
                goals[canon_key(e.kline.signature)] = goal

        def graded(entry: KValue) -> KValue:
            """The ask, graded; every other entry feeds as compiled."""
            base = canon_key(entry.kline.signature)
            if not is_ask(entry.kline.signature) or base not in goals:
                return entry
            goal = goals[base]
            a, b = int(entry.kline.signature), int(goal.kline.signature)
            union = word_atom_count(a | b)
            j = word_atom_count(a & b) / union if union else 1.0
            return KValue(entry.kline, gamma_to_byte(j))

        def is_ask_content(entry: KValue) -> bool:
            """The ask's content form (its canon) — the answer to the
            question, never the feed. It still joins the answering
            pools: the harness releases it when the engine asks."""
            return (
                bool(entry.kline.nodes)
                and not is_ask(entry.kline.signature)
                and canon_key(entry.kline.signature) in goals
            )
        tokens = {
            sig: word
            for sig, word in self._single_token_labels(source).items()
        }
        # Authored sub-scripts: a group is delimited by an entry annotation
        # (or EOF) — a new group opens at any authored (scope-0) entry whose
        # annotation differs from the current group's; '' entries join the
        # current group. MTS entries follow positionally (they carry their
        # scope's annotation), but dedup globally: an MTS entry joins the
        # first group of its annotation, never a later duplicate — the
        # encoder's source-before-MTS partition would otherwise re-open
        # every earlier annotation as a trailing group.
        groups: list[tuple[str, list[KValue]]] = []
        first_by_ann: dict[str, list[KValue]] = {}
        current: list[KValue] | None = None
        current_ann: str | None = None
        for entry in entries:
            annotation = entry.kline.dbg.annotation if entry.kline.dbg else ""
            scope = entry.kline.dbg.scope if entry.kline.dbg else 0
            if scope != 0 and annotation and annotation in first_by_ann:
                first_by_ann[annotation].append(entry)
                continue
            if current is None or (annotation and annotation != current_ann):
                occurrence = sum(
                    1 for k, _ in groups if k.rsplit("#", 1)[0] == annotation
                )
                key = f"{annotation}#{occurrence}"
                groups.append((key, []))
                current = groups[-1][1]
                current_ann = annotation
                if annotation:
                    first_by_ann.setdefault(annotation, current)
            current.append(entry)
        # The answering pools grow as groups open — the harness never answers
        # from a sub-script the dialogue has not reached (no look-ahead).
        heads: dict[int, list[KValue]] = {}
        exact: dict[tuple[int, tuple[int, ...]], KValue] = {}
        words: set[int] = set(tokens)
        # An opening group is a step; its entries join the answering pools
        # when it opens (cumulative — past and current groups only).
        steps: list[tuple[str, list[KValue], KValue]] = []
        for key, group in groups:
            # The opener is the group's question: a scope-0 authored ask,
            # else a scope-0 authored entry, else the canon itself (a bare
            # annotated sig's MTS canon) — never an identity. An identity is
            # an answer, not a question; opening with it grounds the group's
            # words before the question is ever asked.
            opener = next(
                (
                    e for e in group
                    if e.kline.dbg and e.kline.dbg.scope == 0
                    and e.kline.dbg.op == "ASK"
                ),
                next(
                    (e for e in group if e.kline.dbg and e.kline.dbg.scope == 0),
                    next(
                        (
                            e for e in group
                            if e.kline.dbg and e.kline.dbg.op == "CANONICALISES"
                        ),
                        group[0],
                    ),
                ),
            )
            annotation = opener.kline.dbg.annotation if opener.kline.dbg else ""
            if not annotation and not (
                opener.kline.dbg and opener.kline.dbg.op == "ASK"
            ):
                continue
            steps.append((key, group, opener))
        # Scaffold groups open before ask groups: the trainer primes K
        # before asking, so the ask's hops trawl the scaffold from memory.
        # Stable within each class — authored order preserved.
        steps.sort(key=lambda s: bool(s[2].kline.dbg and s[2].kline.dbg.op == "ASK"))
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
            # The opening feeds: "batch" primes K with all the group's
            # scaffolding first, then the entry rationalises against it once
            # grounded (easier for K and for early development). "on-demand"
            # withholds the scaffolding — only the opener enters, and the
            # group's entries answer asks.
            # Terminal words the feeds use whose identities the script
            # never compiled ride along at S1 — without them the words never
            # become known.
            def build_batch(sources: list[KValue]) -> list[KValue]:
                batch = [
                    graded(e) for e in sources
                    if not self.state.is_grounded(e.kline)
                    and not is_ask_content(e)
                ]
                fed = {
                    (e.kline.signature, tuple(e.kline.nodes)) for e in batch
                }
                for entry in sources:
                    for node in entry.kline.nodes:
                        identity = (node, (node,))
                        if (
                            node in words
                            and identity not in fed
                            and not self.state.is_grounded(KLine(node, [node]))
                        ):
                            batch.append(KValue(KLine(node, [node]), SIG_S1))
                            fed.add(identity)
                return batch

            if self.scaffolding == "batch":
                scaffolding = build_batch(
                    [e for e in group if e is not opener]
                )
                if scaffolding:
                    self._drive(step, [scaffolding], heads, exact, words,
                                answered, goals)
                if not self.state.is_grounded(opener.kline):
                    self._drive(step, [[graded(opener)]], heads, exact, words,
                                answered, goals)
            else:
                self._drive(step, [build_batch([opener])], heads, exact,
                            words, answered, goals)
        return results

    def _drive(
        self,
        step: StepResult,
        queue: list[list[KValue]],
        heads: dict[int, list[KValue]],
        exact: dict[tuple[int, tuple[int, ...]], KValue],
        words: set[int],
        answered: set[tuple[int, tuple[int, ...]]],
        goals: dict[int, KValue],
    ) -> None:
        """Run the feed→ask→answer loop until ``queue`` drains."""
        while queue:
            feeds = queue.pop(0)
            before = _grounded_snapshot(self.state)
            batch = self._engine.rationalise(feeds)
            deduped = _dedup(batch)
            after = _grounded_snapshot(self.state)
            grounds = [
                KValue(kl, SIG_S1)
                for key, kl in after.items() if key not in before
            ]
            replies: list[KValue] = []
            turn = Turn(feeds, grounds, deduped)
            step.turns.append(turn)
            for ask_i, ask in enumerate(deduped):
                if self.state.is_grounded(ask.kline):
                    # K stating knowledge it already holds — not a
                    # question. No reply, no escalation.
                    continue
                reply = self._answer(ask, heads, exact, words, answered)
                if reply is None:
                    if is_ask(ask.kline.signature):
                        # An ask is signature discovery, not a
                        # proposal — nothing for a supervisor to decide.
                        replies.append(KValue(ask.kline, SIG_S4))
                        continue
                    graded = self._grade_proposal(ask, goals)
                    if graded is not None:
                        # A proposal under a `==` goal grades at γ of the
                        # two contents — the byte is the trainer's answer,
                        # never the S4 decline.
                        replies.append(graded)
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

    def _grade_proposal(
        self, ask: KValue, goals: dict[int, KValue]
    ) -> KValue | None:
        """A proposal under a ``==`` goal, graded at γ of the proposal's
        content against the goal's target — its signature value (Def 20 —
        fresh depths, so γ = J): the proposal that reached its goal is
        ratified at S1 — the answer just granted; one off the goal grades
        low and refuses on re-feed. ``None`` when no goal pairs with the
        proposal's head."""
        base = canon_key(ask.kline.signature)
        goal = goals.get(base)
        if goal is None or not ask.kline.nodes:
            return None
        a = int(self.signifier.signature_of(ask.kline.nodes))
        b = int(goal.kline.signature)
        union = word_atom_count(a | b)
        j = word_atom_count(a & b) / union if union else 1.0
        return KValue(ask.kline, gamma_to_byte(j))

    def _single_token_labels(self, source: str) -> dict[int, str]:
        """``{signature: word}`` for every single-token word in the source.

        A word that encodes to one typed token is its own signature, so its
        identity ``X:[X]`` is derivable from the tokenizer alone (e.g. an
        abbreviation RHS like ``M(od)`` that never heads a script entry).
        """
        from ks.compiler import Compiler
        from ks.lexer import Lexer
        from ks.parser import Parser
        compiler = Compiler(self._tokenizer, signifier=self.signifier, dev=True,
                            word_bits=self.word_bits,
                            known_words=self.known_words)
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

        An ask-marked kline is answered by the script klines under its
        unmarked base (its canon — the answer), plus an identity when the
        base is a terminal word. A proposal ``A:[B]`` is answered by the
        matching script kline plus its countersignatures.
        """
        kline = ask.kline
        key = (kline.signature, tuple(kline.nodes))
        if key in answered:
            return []
        answered.add(key)
        if is_ask(kline.signature):
            # The ASK marker is not an atom: the question and its canon
            # share the unmarked base — the release pools key on it.
            base = canon_key(kline.signature)
            script_klines = [
                e for e in heads.get(base, [])
                if e.kline.nodes != [base]
                # K already holds it: grounded, or attending to it in the work list
                # (re-feeding the asked question re-arms a refused ask).
                and not self.state.is_grounded(e.kline)
                and not any(
                    entry.signature == e.kline.signature
                    and entry.nodes == e.kline.nodes
                    for entry in self.state.work_list
                )
            ]
            is_word = base in words or any(
                e.kline.nodes == [base]
                for e in heads.get(base, [])
            )
            if not script_klines and not is_word:
                return None
            # Only terminals (single-token words) get a generated identity;
            # a non-terminal's word form must come from the script.
            if not is_word:
                return [*script_klines]
            identity = KValue(KLine(base, [base]), SIG_S1)
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
    tokenizer: BPETokenizer,
    scaffolding: Literal["batch", "on-demand"] = "batch",
) -> Harness:
    """Build a harness over a fresh state: new signifier → state → engine."""
    state = EngineState(NLPSignifier())
    return Harness(tokenizer, Engine(state), scaffolding=scaffolding)


def load_engine(
    path: str | Path,
    tokenizer: BPETokenizer,
    scaffolding: Literal["batch", "on-demand"] = "batch",
) -> Harness:
    """Build a harness over a loaded prior state (reusing its signifier)."""
    signifier = NLPSignifier()
    state = EngineState.load(signifier, path)
    harness = Harness(tokenizer, Engine(state), scaffolding=scaffolding)
    # Compiles must continue the loaded state's word→bit mapping.
    harness.word_bits = dict(state.word_bits or {})
    # And its word binding: the state's words bind chars the script
    # cannot bind itself (the underfit question script's answer chars).
    harness.known_words = list(state.word_bits or {})
    return harness


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
    # Script labels are authoritative; a loaded state's kline label is the
    # fallback for words the current script never mentions.
    name = labels.get(signature) or getattr(signature, "label", "")
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


def _state_labels(state: EngineState) -> dict[int, str]:
    """{value: label} off held klines — prior-lesson MTS forms and word
    bindings the current script never names."""
    out: dict[int, str] = {}

    def take(v) -> None:
        label = getattr(v, "label", "")
        if label:
            out.setdefault(int(v), label)

    for store in (state.frame, state.ltm):
        for bucket in store.values():
            for k in bucket:
                take(k.signature)
                for n in k.nodes:
                    take(n)
    for k in state.work_list:
        take(k.signature)
        for n in k.nodes:
            take(n)
    return out


def _sig_to_label(source: str, tokenizer: BPETokenizer, signifier: NLPSignifier,
                  word_bits: dict[str, int] | None = None,
                  state: EngineState | None = None) -> dict[int, str]:
    """Recompile once to recover ``{signature: scripted label}`` for display.

    Compiled-entry labels are authoritative; the encoder's ``node_labels``
    supplies words for single-token node values that never head an entry
    (e.g. MTS-expanded words like ``did``/``have``).
    """
    from ks.compiler import Compiler
    from ks.lexer import Lexer
    from ks.parser import Parser
    compiler = Compiler(tokenizer, signifier=signifier, dev=True, word_bits=word_bits)
    entries = compiler.compile(Parser(Lexer(source).tokenize()).parse())
    out: dict[int, str] = dict(compiler.node_labels)
    for e in entries:
        d = e.kline.dbg
        if d is None:
            continue
        label = d.label
        if label:
            out.setdefault(e.kline.signature, label)
    if state is not None:
        for value, label in _state_labels(state).items():
            out.setdefault(value, label)
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


def _grounded_snapshot(state: EngineState) -> dict[tuple[KNode, tuple[KNode, ...]], KLine]:
    """Every grounded kline in frame and LTM, keyed by (signature, nodes)."""
    snap: dict[tuple[KNode, tuple[KNode, ...]], KLine] = {}
    for store in (state.frame, state.ltm):
        for signature, bucket in store.items():
            for kl in bucket:
                snap.setdefault((signature, tuple(kl.nodes)), kl)
    return snap


def _render_grounded(state: EngineState, labels: dict[int, str], verbose: bool,
                        pre_grounded: set[tuple[KNode, tuple[KNode, ...]]] | None = None) -> str:
    if not state.ltm:
        return "  (grounded nothing)"
    lines = []
    reloaded: list[str] = []
    for signature in sorted(state.ltm, key=lambda s: (s.bit_length(), s)):
        bucket = state.ltm[signature]
        for kl in bucket:
            nodes = ", ".join(_label(n, labels, verbose) for n in kl.nodes)
            line = f"      {_label(kl.signature, labels, verbose)}:[{nodes}]"
            if pre_grounded is not None and (signature, tuple(kl.nodes)) in pre_grounded:
                reloaded.append(line)
            else:
                lines.append(line)
    if pre_grounded is not None:
        lines.append("    (reloaded, held before this run)")
        lines.extend(reloaded)
    return "\n".join(lines)


def _render_frame(state: EngineState, labels: dict[int, str], verbose: bool) -> str:
    """Frame entries with no isomorphic (signature, nodes) entry in LTM."""
    lines: list[str] = []
    for signature in sorted(state.frame, key=lambda s: (s.bit_length(), s)):
        for kl in state.frame[signature]:
            if any(
                existing.nodes == kl.nodes
                for existing in state.ltm.get(signature, [])
            ):
                continue
            nodes = ", ".join(_label(n, labels, verbose) for n in kl.nodes)
            lines.append(f"      {_label(kl.signature, labels, verbose)}:[{nodes}]")
    return "\n".join(lines) if lines else "  (empty)"


def _render_work_list(state: EngineState, labels: dict[int, str], verbose: bool) -> str:
    if not state.work_list:
        return "  (empty)"
    return "\n".join(
        f"      {_render_kline_struct(kl, labels, verbose)}" for kl in state.work_list
    )


# ── Model graph ───────────────────────────────────────────────────────────

_GRAPH_LAYERS = ("L", "F", "W", "R")  # ltm, frame, work_list, refused

# ANSI stand-ins for the renderer fills (ltm green, frame yellow, work
# blue, refused red); enabled on a tty unless NO_COLOR is set.
_ANSI = {"L": "\033[32m", "F": "\033[33m", "W": "\033[34m", "R": "\033[31m"}
_COLOUR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _paint(text: str, layers: str | set[str]) -> str:
    if not _COLOUR:
        return text
    return f"{_ANSI[min(layers, key=_GRAPH_LAYERS.index)]}{text}\033[0m"


def _structure_class(kline: KLine, signifier: NLPSignifier) -> str:
    if is_unknown(kline):
        return "unknown"
    if is_identity(kline):
        return "id"
    if is_canon(kline, signifier):
        return "canon"
    if is_relationship(kline):
        # Case 4 vs case 6 by coverage (overlap); the covered shapes with
        # excess (A:[AB], AB:[BC]) fall through to the multi-node classes.
        if is_denotation(kline, signifier):
            return "denotation"
        if is_connotation(kline, signifier):
            return "connotation"
    under, over = classify_misfit(kline, signifier)
    if under and not over:
        return "under"
    if over and not under:
        return "over"
    return "misfit"


def _model_graph(
    state: EngineState,
) -> tuple[dict[tuple[KNode, tuple[KNode, ...]], tuple[str, KLine]],
           dict[KNode, set[str]]]:
    """The model across all layers as ``(klines, values)``.

    ``klines`` maps each distinct (signature, nodes) to the layer glyphs
    holding it plus a representative kline (ltm → frame → work list →
    refused). ``values`` maps each value to the layers it appears in, in
    any role.
    """
    klines: dict[tuple[KNode, tuple[KNode, ...]], tuple[str, KLine]] = {}

    def add(kl: KLine, glyph: str) -> None:
        key = (kl.signature, tuple(kl.nodes))
        held, _ = klines.get(key, ("", kl))
        klines[key] = (held + glyph, kl)

    for bucket in state.ltm.values():
        for kl in bucket:
            add(kl, "L")
    for bucket in state.frame.values():
        for kl in bucket:
            add(kl, "F")
    for kl in state.work_list:
        add(kl, "W")
    for sig, nodes in state.refused:
        add(KLine(sig, list(nodes)), "R")

    values: dict[KNode, set[str]] = {}
    for (sig, nodes), (layers, _) in klines.items():
        values.setdefault(sig, set()).update(layers)
        for n in nodes:
            values.setdefault(n, set()).update(layers)
    return klines, values


def _graph_name(node: KNode, labels: dict[int, str], verbose: bool, signifier = None) -> str:
    return _label(node, labels, verbose)


def _sorted_values(values: dict[KNode, set[str]]) -> list[KNode]:
    return sorted(values, key=lambda n: (n.bit_length(), n))


def _compound_of(kline: KLine, signifier: NLPSignifier) -> KNode | None:
    """The composed node of a multi-node kline — the OR-reduction of its
    nodes, labelled by their concatenation (e.g. SubjectVerbObject)."""
    if len(kline.nodes) < 2:
        return None
    return signifier.signature_of(kline.nodes)


def _compound_name(compound: KNode, verbose: bool) -> str:
    # The composed label (node labels concatenated), never the script label
    # a same-valued signature may carry — the compound names what composes it.
    name = compound.label or f"0x{int(compound):x}"
    return f"{name}|0x{int(compound):x}" if verbose else name


def _graph_heads(
    klines: dict[tuple[KNode, tuple[KNode, ...]], tuple[str, KLine]],
) -> tuple[set[KNode], dict[KNode, str], set[KNode]]:
    """Values heading a non-identity kline, held identities by layer glyphs,
    and values appearing as nodes of multi-node klines."""
    heads = {
        kl.signature for (_, _), (_, kl) in klines.items() if not is_identity(kl)
    }
    identities = {
        kl.signature: layers
        for (_, _), (layers, kl) in klines.items()
        if is_identity(kl)
    }
    compound_members = {
        n
        for (_, _), (_, kl) in klines.items()
        if len(kl.nodes) > 1
        for n in kl.nodes
    }
    return heads, identities, compound_members


def _render_model_graph(state: EngineState, labels: dict[int, str],
                        verbose: bool) -> str:
    signifier = state.signifier
    klines, values = _model_graph(state)
    if not values:
        return "── model graph ──\n  (empty)"
    heads, identities, compound_members = _graph_heads(klines)
    by_head: dict[KNode, list[tuple[str, KLine]]] = {}
    for (sig, _nodes), (layers, kl) in klines.items():
        by_head.setdefault(sig, []).append((layers, kl))
    key = "  ".join(
        _paint(f"● {n}", g)
        for g, n in (("L", "ltm"), ("F", "frame"), ("W", "work"), ("R", "refused"))
    )
    lines = [
        f"── model graph ──  {key}  *=heads klines of its own",
        "    multi-node klines: sig ---> compound ---> identity-held members",
    ]
    leaves: list[str] = []
    for node in _sorted_values(values):
        name = _graph_name(node, labels, verbose)
        if node in heads:
            lines.append(f"  {_paint(name, values[node])}")
            seen_compounds: set[int] = set()
            for klayers, kl in by_head[node]:
                if is_identity(kl):
                    continue
                cls = _structure_class(kl, signifier)
                compound = _compound_of(kl, signifier)
                if compound is not None:
                    if int(compound) in seen_compounds:
                        continue
                    seen_compounds.add(int(compound))
                    lines.append(
                        f"      {cls:<11} {_paint('--->', klayers)} "
                        f"{_compound_name(compound, verbose)}"
                    )
                    for n in kl.nodes:
                        if n not in identities:
                            continue
                        lines.append(
                            f"{' ':27}{_paint('--->', identities[n])} "
                            f"{_graph_name(n, labels, verbose)}"
                            f"{'*' if n in heads else ''}"
                        )
                elif kl.nodes:
                    n = kl.nodes[0]
                    lines.append(
                        f"      {cls:<11} {_paint('--->', klayers)} "
                        f"{_graph_name(n, labels, verbose)}"
                        f"{'*' if n in heads else ''}"
                    )
                else:
                    lines.append(f"      {cls:<11} {_paint('--->', klayers)} ∅")
        elif node in identities and node not in compound_members:
            lines.append(f"  {_paint(name, values[node])}")
        elif node not in identities:
            leaves.append(f"  {_paint(name, values[node])}")
            continue
        else:
            continue
        if node in identities and node not in compound_members:
            lines.append(f"      {'id':<11} [{_paint(name, identities[node])}]")
    if leaves:
        lines.append("  -- referenced, never headed --")
        lines.extend(leaves)
    return "\n".join(lines)


def _graph_fill(layers: str | set[str]) -> str:
    strongest = min(layers, key=_GRAPH_LAYERS.index)
    return {
        "L": "#d5e8d4", "F": "#fff2cc", "W": "#dae8fc", "R": "#f8cecc",
    }[strongest]


def _graph_class(layers: str | set[str]) -> str:
    return {_GRAPH_LAYERS[i]: name
            for i, name in enumerate(("ltm", "frame", "work", "refused"))}[
        min(layers, key=_GRAPH_LAYERS.index)
    ]


def _render_model_graph_dot(state: EngineState, labels: dict[int, str],
                            verbose: bool) -> str:
    signifier = state.signifier
    klines, values = _model_graph(state)
    heads, identities, compound_members = _graph_heads(klines)

    def nid(n: KNode) -> str:
        return f"n{int(n):x}"

    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        "digraph model {",
        "  rankdir=BT;",
        '  graph [labelloc=b, '
        'label="green=ltm  yellow=frame  blue=work  red=refused"];',
        '  node [shape=ellipse, style=filled, fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]
    for node in _sorted_values(values):
        name = esc(_graph_name(node, labels, verbose, signifier))
        lines.append(
            f'  {nid(node)} [label="{name}", '
            f'fillcolor="{_graph_fill(values[node])}"];'
        )
    compounds: dict[int, KNode] = {}
    for (_, _), (_, kl) in klines.items():
        compound = _compound_of(kl, signifier)
        if compound is not None:
            compounds.setdefault(int(compound), compound)
    for value, compound in sorted(compounds.items()):
        lines.append(
            f'  c{value:x} [label="'
            f'{esc(_compound_name(compound, verbose))}", '
            f'shape=box, style="filled,dashed", fillcolor="#ffffff"];'
        )
    edge_style = {
        "canon": "",
        "under": ' style="dashed" penwidth="2"',
        "over": ' style="dashed" penwidth="2"',
        "misfit": ' style="dashed" penwidth="2"',
        "connotation": ' style="dashed"',
        "denotation": ' style="dashed"',
        "identity": ' style="dotted"',
        "unknown": "",
    }
    member_edges: set[tuple[int, int]] = set()
    compound_edges: set[tuple[int, int]] = set()
    for (_sig, _nodes), (_layers, kl) in klines.items():
        if is_identity(kl):
            if kl.signature not in compound_members:
                lines.append(
                    f"  {nid(kl.signature)} -> {nid(kl.signature)}"
                    f"[{edge_style['identity']}];"
                )
            continue
        cls = _structure_class(kl, signifier)
        style = edge_style[cls]
        compound = _compound_of(kl, signifier)
        if compound is not None:
            edge_key = (int(kl.signature), int(compound))
            if edge_key not in compound_edges:
                compound_edges.add(edge_key)
                lines.append(
                    f'  {nid(kl.signature)} -> c{edge_key[1]:x} '
                    f'[label="{cls}"{style}];'
                )
            for n in kl.nodes:
                key = (int(compound), int(n))
                if n in identities and key not in member_edges:
                    member_edges.add(key)
                    lines.append(
                        f"  c{key[0]:x} -> {nid(n)}"
                        f"[{edge_style['identity']}];"
                    )
        else:
            for n in kl.nodes:
                lines.append(
                    f'  {nid(kl.signature)} -> {nid(n)} [label="{cls}"{style}];'
                )
    lines.append("}")
    return "\n".join(lines)


def _render_model_graph_mermaid(state: EngineState, labels: dict[int, str],
                                verbose: bool) -> str:
    signifier = state.signifier
    klines, values = _model_graph(state)
    heads, identities, compound_members = _graph_heads(klines)

    def nid(n: KNode) -> str:
        return f"n{int(n):x}"

    lines = ["flowchart BT"]
    for cls, fill in (("ltm", "#d5e8d4"), ("frame", "#fff2cc"),
                      ("work", "#dae8fc"), ("refused", "#f8cecc")):
        lines.append(f"  classDef {cls} fill:{fill}")
    lines.append("  classDef compound fill:#ffffff,stroke-dasharray: 5 5")
    for node in _sorted_values(values):
        name = _graph_name(node, labels, verbose, signifier).replace('"', "")
        lines.append(
            f'  {nid(node)}("{name}")'
            f":::{_graph_class(values[node])}"
        )
    compounds: dict[int, KNode] = {}
    for (_, _), (_, kl) in klines.items():
        compound = _compound_of(kl, signifier)
        if compound is not None:
            compounds.setdefault(int(compound), compound)
    for value, compound in sorted(compounds.items()):
        name = _compound_name(compound, verbose).replace('"', "")
        lines.append(f'  c{value:x}("{name}"):::compound')
    member_edges: set[tuple[int, int]] = set()
    compound_edges: set[tuple[int, int]] = set()
    for (_sig, _nodes), (layers, kl) in klines.items():
        if is_identity(kl):
            if kl.signature not in compound_members:
                lines.append(f"  {nid(kl.signature)} --> {nid(kl.signature)}")
            continue
        cls = _structure_class(kl, signifier)
        compound = _compound_of(kl, signifier)
        if compound is not None:
            edge_key = (int(kl.signature), int(compound))
            if edge_key not in compound_edges:
                compound_edges.add(edge_key)
                lines.append(
                    f'  {nid(kl.signature)} -->|"{cls}"| c{edge_key[1]:x}'
                )
            for n in kl.nodes:
                key = (int(compound), int(n))
                if n in identities and key not in member_edges:
                    member_edges.add(key)
                    lines.append(f"  c{key[0]:x} --> {nid(n)}")
        else:
            for n in kl.nodes:
                lines.append(f'  {nid(kl.signature)} -->|"{cls}"| {nid(n)}')
    return "\n".join(lines)


_GRAPH_RENDERERS = {
    "ascii": _render_model_graph,
    "dot": _render_model_graph_dot,
    "mermaid": _render_model_graph_mermaid,
}


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
        f"  frame (framed, not yet grounded):\n{_render_frame(state, labels, verbose)}\n"
        f"  work_list (attending to at end of run):\n{_render_work_list(state, labels, verbose)}"
    )


def present(results: list[StepResult], state: EngineState, source: str,
            tokenizer: BPETokenizer, signifier: NLPSignifier, *, verbose: bool,
            pre_grounded: set[tuple[KNode, tuple[KNode, ...]]] | None = None,
            word_bits: dict[str, int] | None = None,
            graph: Literal["ascii", "dot", "mermaid"] | None = None) -> None:
    labels = _sig_to_label(source, tokenizer, signifier, word_bits, state)
    last_annotation: str | None = None
    for step in results:
        annotation = step.entry.kline.dbg.annotation if step.entry.kline.dbg else ""
        if annotation and annotation != last_annotation:
            print(f"\n[{annotation}]")
            last_annotation = annotation
        print(_render_step(step, labels, verbose))
    print()
    print(_render_summary(results, state, labels, verbose, pre_grounded))
    if graph is not None:
        print()
        print(_GRAPH_RENDERERS[graph](state, labels, verbose))


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
             "engine and present the trace.",
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
             "evidence (canons, countersigns, connotations) decides the band, "
             "S1 only when the script's proof completes.",
    )
    parser.add_argument(
        "--scaffolding", choices=("batch", "on-demand"), default="batch",
        help="How a group's scaffolding reaches the engine: 'batch' (default) "
             "delivers all compiled klines with the opening entry; 'on-demand' "
             "feeds only the opener and withholds scaffolding until K asks.",
    )
    parser.add_argument(
        "-g", "--graph", nargs="?", const="ascii", default=None,
        choices=("ascii", "dot", "mermaid"),
        help="After the summary, print a graph of the end-of-run model state "
             "across all layers (L ltm, F frame, W work list, R refused): "
             "'ascii' in the terminal (the default when bare), 'dot' as "
             "Graphviz DOT, 'mermaid' as a Mermaid flowchart.",
    )

    parser.add_argument(
        "-p", "--persist", nargs="?", const="auto", default=None, metavar="PATH",
        help="Load engine state before the run and save it after. PATH "
             "defaults to data/dialogue/{script_name}.json. Also writes a "
             "graph of the end-of-run state next to the saved file (as .dot, "
             "or .mmd with --graph mermaid) for VSCode preview extensions.",
    )
    args = parser.parse_args(argv)

    source_path = Path(args.source)
    if source_path.suffix == ".ks":
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"harness: could not read {args.source!r}: {exc}", file=sys.stderr)
            return 2
        tok = BPETokenizer()
        scaffolding = cast(Literal["batch", "on-demand"], args.scaffolding)
        state_path = (
            Path(f"data/dialogue/{source_path.stem}.json")
            if args.persist == "auto"
            else Path(args.persist) if args.persist else None
        )
        if state_path is not None and state_path.exists():
            harness = load_engine(state_path, tok, scaffolding=scaffolding)
            n = sum(len(b) for b in harness.state.ltm.values())
            print(f"── running on reloaded state: {n} grounded klines "
                  f"from {state_path} ──")
        else:
            harness = make_engine(tok, scaffolding=scaffolding)
        if args.structural:
            from dialogue.structural import SemanticEvidence, StructuralSupervisor

            labels = _sig_to_label(source, tok, harness.signifier,
                                   harness.word_bits, harness.state)
            render = lambda v: _render_kline(v, labels, args.verbose)  # noqa: E731
            supervisor = StructuralSupervisor(
                SemanticEvidence(harness.signifier), harness.state, render
            )
            harness._escalate = supervisor
        if args.supervise:
            labels = _sig_to_label(source, tok, harness.signifier,
                                   harness.word_bits, harness.state)
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
                verbose=args.verbose, pre_grounded=pre_grounded,
                word_bits=harness.word_bits, graph=args.graph)
        if state_path is not None:
            harness.state.word_bits = harness.word_bits
            # Always save state to script named path
            state_path = Path(f"data/dialogue/{source_path.stem}.json")
            harness.state.save(state_path)
            fmt = args.graph if args.graph in ("dot", "mermaid") else "dot"
            graph_path = state_path.with_suffix(".dot" if fmt == "dot" else ".mmd")
            labels = _sig_to_label(source, tok, harness.signifier,
                                   harness.word_bits, harness.state)
            graph_path.write_text(
                _GRAPH_RENDERERS[fmt](harness.state, labels, args.verbose)
            )
            print(f"── model graph written to {graph_path} ──")
        return 0

    try:
        document = CurriculumDocument.from_file(source_path)
    except (CurriculumParseError, OSError) as exc:
        print(f"harness: could not read source {args.source!r}: {exc}", file=sys.stderr)
        return 2

    tok = BPETokenizer()
    harness = make_engine(tok)
    cumulative = ""
    for lesson in document.lessons:
        source = "\n".join(lesson.kscript)
        cumulative = f"{cumulative}\n{source}"
        print(f"\n══ lesson {lesson.label} ══")
        results = harness.run(source)
        present(results, harness.state, cumulative, tok, harness.signifier,
                verbose=args.verbose, graph=args.graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
