"""The dialogue harness.

A minimal, synchronous, non-judging loop. Compiles a KScript source and
runs it as a sequence of ask==goal scenarios: every `==` ask whose named
goal resolves to a kline entry is one scenario — the ask presented once,
the goal handed to it directly, never presented and never stored. Every
other kline is held as scaffolding, not primed: the domain starts empty,
and a stall demands — the harness applies the candidates logic to what
it holds, from the stalled entry's seat, adds what qualifies to the
domain, and the model retries (the stalled component pushed back on
the stack). Scenarios run in ask order over the one live domain. Priming is a harness fact — the harness owns the
domain, and nothing learns into it during a run: the Cogitator reads it,
stores nothing, and broadcasts its claims to the harness (the claim sink)
at S2. Claims are reported; they persist nowhere, for now.

Each step's trace table prints by default; the end-of-run state (the
held domain, the open proposals) only with ``-v``.

Usage::

    PYTHONPATH=src:. python -m dev.dialogue.harness path/to/script.ks
    PYTHONPATH=src:. python -m dev.dialogue.harness path/to/script.ks -v
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.kline import KNode, is_ask, mts, using_word_labels
from kalvin.kvalue import KValue
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source, path_resolver

from rationalisation import (
    Domain,
    Event,
    KLine,
    Entry,
    Rationalisation,
    Cogitator,
    band,
    can,
    effort,
    candidates
)

# The appendix's content mask: the word half is content; the token half is
# inert encoding cargo, masked off at every content read. The model
# operates in content space — values are masked once, here at the boundary.
WORD_MASK = 0x7FFF_FFFF_0000_0000


@dataclass
class Step:
    """One presented entry and what its rationalisation produced."""

    index: int
    entry: KValue  # the compiled entry (annotation/label live here)
    kline: KLine  # the model form
    result: Rationalisation

class Runner:
    """Compile → present → rationalise → record, over one live Domain."""

    def __init__(self, tokenizer: BPETokenizer | None = None) -> None:
        self.tokenizer = tokenizer or BPETokenizer()
        self.signifier = NLPSignifier()
        self.domain = Domain()
        self.scaffolding = Domain()  # held klines, served on demand at a stall
        self.word_bits: dict[str, int] = {}
        self.claims: list[tuple[int, KLine]] = []  # (step, claim) — appended at grounding, mid-run
        self.stalls: list[tuple[int, KLine]] = []  # (step, stalled kline) — appended at run's end
        self._step_index = -1  # the presenting step — claims and stalls arrive before it completes
        self.cogitator = Cogitator(self.domain, on_claim=self._claim, on_stall=self._stall)

    def _claim(self, claim: KLine) -> None:
        """The claim sink — a claim lands the moment it grounds, not at return."""
        self.claims.append((self._step_index, claim))
        # self.domain.add(claim)

    def _stall(self, kline: KLine, entry: Entry) -> bool:
        """The stall sink — an S4 stall demands scaffolding: the held klines
        that qualify as candidates for the stalled signature, from the
        entry's own seat (the same logic the model reads D with), join the
        domain and the model retries; an unserved stall files itself as a
        connect proposal: fit wanted, nobody to talk to."""
        cands = candidates(kline.signature, entry, self.scaffolding)
        if not cands:
            self.stalls.append((self._step_index, kline))
            return False
        for k in cands:
            self.domain.add(k)
        return True

    def run(self, source: str, resolver=None, scaffold=False) -> list[Step]:
        """An ask==goal run: every `==` ask with a resolvable goal is one
        scenario — the ask presented once, the goal handed to it directly
        (never presented, never stored). Everything else primes the domain;
        scenarios run in ask order over the one live domain."""
        entries = compile_source(
            source,
            tokenizer=self.tokenizer,
            signifier=self.signifier,
            dev=True,
            word_bits=self.word_bits,
            resolver=resolver,
        )
        pairs = self._ask_goal_pairs(entries)
        if not pairs:
            raise ValueError("ask==goal: no `==` ask with a findable goal kline")
        paired = {id(e) for pair in pairs for e in pair}
        # The word registry — derived values (fits, residuals) label from
        # their component words while the run is live.
        words = {(bit << 32): word for word, bit in self.word_bits.items()}
        with using_word_labels(words):
            for other in entries:
                if id(other) not in paired:
                    k = _to_kline(other)
                    if scaffold:
                        self.scaffolding.add(k)  # held, never an R1 store
                    else:
                        self.domain.add(k)  # held, never an R1 store
            steps: list[Step] = []
            for ask, goal in pairs:
                self._step(steps, len(steps), ask, goal=_to_kline(goal))
        return steps

    @staticmethod
    def _ask_goal_pairs(entries: list[KValue]) -> list[tuple[KValue, KValue]]:
        """Every `==` ask whose named goal resolves to a kline entry, in ask
        order — each pair one scenario; an imported module's ask precedes
        the importing script's. The goal is the last label match (the fullest
        authored definition): has nodes, not itself an ask."""
        pairs: list[tuple[KValue, KValue]] = []
        for e in entries:
            d = e.kline.dbg
            if d is None or d.op != "ASK" or not d.goal:
                continue
            goal = None
            for g in entries:
                if (
                    g.kline.dbg
                    and g.kline.dbg.label == d.goal
                    and g.kline.nodes
                    and not is_ask(g.kline.signature)
                ):
                    goal = g
            if goal is not None:
                pairs.append((e, goal))
        return pairs

    def _step(self, steps: list[Step], i: int, value: KValue, goal: KLine) -> None:
        """Present one entry: rationalise, record, discharge. Claims and
        stalls arrive through the sinks while the run is live."""
        kline = _to_kline(value)
        self._step_index = i
        def end_step(result: Rationalisation):
            steps.append(Step(i, value, kline, result))

        self.cogitator.rationalise(kline, goal, end_step)
        self.cogitator.cogitate()

def _to_kline(value: KValue) -> KLine:
    """The compiled entry as a content-space kline (appendix: cargo masked,
    labels preserved — nodes stay KNode throughout, authored order and
    repetitions kept)."""
    sig = value.kline.signature
    masked_sig = KNode(int(sig) & WORD_MASK, getattr(sig, "label", ""))
    nodes = [
        KNode(int(n) & WORD_MASK, getattr(n, "label", "")) for n in value.kline.nodes
    ]
    return KLine(masked_sig, nodes)


# ── Labels ────────────────────────────────────────────────────────────────


def _sig_to_label(source: str, tokenizer: BPETokenizer, signifier: NLPSignifier,
                  word_bits: dict[str, int] | None = None,
                  resolver: Callable[[str], str] | None = None) -> dict[int, str]:
    """Recompile once to recover ``{signature: scripted label}`` for display.

    Compiled-entry labels are authoritative; the encoder's ``node_labels``
    supplies words for single-token node values that never head an entry
    (e.g. compound-expanded words like ``did``/``have``).
    """
    from ks.compiler import Compiler
    from ks.lexer import Lexer
    from ks.parser import Parser
    compiler = Compiler(tokenizer, signifier=signifier, dev=True, word_bits=word_bits,
                        resolver=resolver)
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


# ── Presentation ──────────────────────────────────────────────────────────


def _cover(v: int, labels: dict[int, str], alloc: dict[int, str]) -> list[str]:
    """Decompose a derived value into labelled parts (display only — the
    algebra never looks inside a node). Whole known signatures cover first,
    then the one-bit-per-word allocation region."""
    parts: list[str] = []
    rest = v
    for sig in sorted(labels, key=lambda s: (s.bit_count(), s), reverse=True):
        if sig and sig & rest == sig:
            parts.append(labels[sig])
            rest &= ~sig
    for bit in alloc:  # allocation order — authored word order
        if bit & rest == bit:
            parts.append(alloc[bit])
            rest &= ~bit
    if rest:
        parts.append(f"+0x{rest:x}")
    return parts


def _label(v: int, labels: dict[int, str], alloc: dict[int, str], verbose: bool) -> str:
    name = labels.get(v) or getattr(v, "label", "")
    if not name:
        # A derived value (an R2 residual, a substituted canon): compose
        # its label from the words it covers.
        name = "|".join(_cover(v, labels, alloc)) if v else "⊥"
    if verbose:
        return f"{name}|0x{v:x}"
    return name


def _render(k: KLine, labels: dict[int, str], alloc: dict[int, str], verbose: bool) -> str:
    nodes = ", ".join(_label(n, labels, alloc, verbose) for n in k.nodes)
    return f"{_label(k.signature, labels, alloc, verbose)}:[{nodes}]"


def _words(v: int, labels: dict[int, str], alloc: dict[int, str]) -> list[str]:
    """The word decomposition of a value — authored word order, no whole-signature
    preference (the table spells regions out)."""
    if not v:
        return []
    parts: list[str] = []
    rest = int(v)
    for bit in sorted(alloc):  # allocation order — authored word order
        if bit & rest == bit:
            parts.append(alloc[bit])
            rest &= ~bit
    if rest:
        parts.append(f"+0x{rest:x}")
    return parts


def _humps(name: str) -> list[str]:
    """The camelCase parts of a label (acronyms kept whole)."""
    return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", name) or [name]


def _tlabel(v: int, labels: dict[int, str], alloc: dict[int, str]) -> str:
    """A table label: a value that carries a label is named by it (MTS
    from the mint, initials from the word registry); the script registry
    fills the gaps, and provenance expressions (∖, ∧) fall through to the
    word decomposition."""
    if not v:
        return ""
    name = getattr(v, "label", "") or labels.get(int(v))
    if name and "∖" not in name and "∧" not in name:
        return mts(_humps(name))
    return mts(_words(v, labels, alloc))


def _tframe(v: KNode, labels: dict[int, str], alloc: dict[int, str]) -> str:
    """Frame/event content in MTS form."""
    return _tlabel(v, labels, alloc)


def _event_line(ev: Event, labels: dict[int, str], alloc: dict[int, str]) -> list[str]:
    c = ""
    if ev.candidate is not None:
        c = _tlabel(ev.candidate.signature, labels, alloc)
        if ev.candidate.nodes and can(ev.candidate.nodes) != ev.candidate.signature:
            c += ":" + " ".join(_tlabel(n, labels, alloc) for n in ev.candidate.nodes)
    kind = "" if ev.kind == "entry" else ev.kind
    return [
        _tlabel(ev.entry, labels, alloc),
        c,
        _tlabel(ev.F, labels, alloc),
        _tlabel(ev.U, labels, alloc),
        _tlabel(ev.O, labels, alloc),
        _tframe(ev.frame, labels, alloc) if ev.frame else "",
        kind,
    ]


def _render_events(events: list[Event], labels: dict[int, str], alloc: dict[int, str]) -> list[str]:
    """The event table: ask, candidate, fit, underfit, overfit,
    frame, event — one line per event, in order."""
    hdr = ["A(sk)", "C(andidate)", "F(it)", "U(nderfit)", "O(verfit)", "frame", "event"]
    body = [_event_line(ev, labels, alloc) for ev in events]
    widths = [
        max(len(hdr[i]), *(len(r[i]) for r in body)) if body else len(hdr[i])
        for i in range(len(hdr))
    ]
    lines = [
        "| " + " | ".join(hdr[i].ljust(widths[i]) for i in range(len(hdr))) + " |",
        "|" + "|".join("-" * (widths[i] + 2) for i in range(len(hdr))) + "|",
    ]
    for r in body:
        lines.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(hdr))) + " |")
    return lines


def _tline(k: KLine, labels: dict[int, str], alloc: dict[int, str]) -> str:
    """A kline in table form — sig in MTS form, nodes spelled out."""
    sig = _tlabel(k.signature, labels, alloc)
    nodes = ", ".join(_tlabel(n, labels, alloc) for n in k.nodes)
    return f"{sig}:[{nodes}]" if k.nodes else f"{sig}:[]"


def _render_step(step: Step, claims: list[KLine], stalled: list[KLine], labels: dict[int, str], alloc: dict[int, str], verbose: bool) -> str:
    lines = [
        f"── Step {step.index + 1}  {band(step.kline).value}  "
        f"{_render(step.kline, labels, alloc, verbose)} ──"
    ]
    if step.result.events and not claims:
        lines.append("      (no claim — every event below)")
    tallies: list[tuple[KLine, int]] = []  # distinct claims, counted
    for c in claims:
        for i, (s, n) in enumerate(tallies):
            if s == c:
                tallies[i] = (s, n + 1)
                break
        else:
            tallies.append((c, 1))
    lines.extend("      " + l for l in _render_events(step.result.events, labels, alloc))
    for c, n in tallies:
        count = f"  ×{n}" if n > 1 else ""
        lines.append(f"      claim     {_tline(c, labels, alloc)}{count}")
    exemplars = stalled[:2]
    for k in exemplars:
        lines.append(f"      stall     S4 no candidates — {_tline(k, labels, alloc)}  E={effort(k)}")
    if len(stalled) > len(exemplars):
        lines.append(f"      stall     S4 no candidates ×{len(stalled)} (+{len(stalled) - len(exemplars)} more)")
    return "\n".join(lines)


def present(
    steps: list[Step],
    domain: Domain,
    claims: list[tuple[int, KLine]],
    stalls: list[tuple[int, KLine]],
    source: str,
    tokenizer: BPETokenizer,
    signifier: NLPSignifier,
    *,
    verbose: bool,
    word_bits: dict[str, int] | None = None,
    resolver: Callable[[str], str] | None = None,
) -> None:
    """Each step's trace table; with ``verbose``, the end-of-run state too —
    the held domain and the open proposals."""
    labels = _sig_to_label(source, tokenizer, signifier, word_bits, resolver=resolver)
    # Script labels key on full compiled values; content-space values need
    # their labels carried over under the mask.
    content_labels: dict[int, str] = {}
    for sig, label in labels.items():
        masked = sig & WORD_MASK
        if masked and masked not in content_labels:
            content_labels[masked] = label
    labels = {**content_labels, **labels}
    # The one-bit-per-word allocation region — how derived values decompose.
    alloc = {(bit << 32): word for word, bit in (word_bits or {}).items()}
    last_annotation: str | None = None
    for step in steps:
        annotation = step.entry.kline.dbg.annotation if step.entry.kline.dbg else ""
        if annotation and annotation != last_annotation:
            print(f"\n[{annotation}]")
            last_annotation = annotation
        claimed = [c for j, c in claims if j == step.index]
        stalled = [k for j, k in stalls if j == step.index]
        print(_render_step(step, claimed, stalled, labels, alloc, verbose))
    if not verbose:
        return
    print("\n── domain (held at end of run) ──")
    for k in sorted(domain, key=lambda x: (x.signature.bit_length(), x.signature)):
        mark = "canon " if can(k.nodes) == k.signature else "claim "
        print(f"      {mark}{_render(k, labels, alloc, verbose)}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a .ks script through the rationalisation model.")
    parser.add_argument("source", help="Path to a .ks KScript file")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show the end-of-run state (held domain, open proposals) and hex "
             "signatures alongside labels.",
    )
    parser.add_argument(
        "-s", "--scaffold", action="store_true",
        help="Hold scaffolding prior to run",
    )
    args = parser.parse_args(argv)

    path = Path(args.source)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"harness: could not read {args.source!r}: {exc}", file=sys.stderr)
        return 2
    runner = Runner()
    try:
        steps = runner.run(source, resolver=path_resolver(path.parent), scaffold=args.scaffold)
    except ValueError as exc:
        print(f"harness: {exc}", file=sys.stderr)
        return 2
    present(
        steps, runner.domain, runner.claims, runner.stalls, source, runner.tokenizer,
        runner.signifier, verbose=args.verbose, word_bits=runner.word_bits,
        resolver=path_resolver(path.parent),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
