"""The dialogue harness.

A minimal, synchronous, non-judging loop. Compiles a KScript source and
runs it as a sequence of ask==goal scenarios: every `==` ask whose named
goal resolves to a kline entry is one scenario — the ask authored first,
the goal answering — and every other kline primes the domain before
rationalisation. Scenarios run in ask order over one live domain; an
earlier scenario's stores and proposals persist into the later ones.
Priming is a
harness fact, never a model admission: R1 remains the sole way the run
itself learns. Nothing is checked at presentation; unheld references
find no neighbours.

Proposals persist across steps: a later presented entry discharges a
pending proposal when its content overlaps the target (Definition 15).
The presenter reports; it never judges.

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
from kalvin.kline import KNode, is_ask
from kalvin.kvalue import KValue
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source, path_resolver

from rationalisation import (
    Domain,
    End,
    Event,
    Kline,
    Proposal,
    Rationalisation,
    Cogitator,
    band,
    can,
    proposals,
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
    kline: Kline  # the model form
    result: Rationalisation


@dataclass
class Pending:
    """A proposal awaiting an answer, and its fate."""

    prop: Proposal
    raised_at: int
    discharged_at: int | None = None


class Runner:
    """Compile → present → rationalise → record, over one live Domain."""

    def __init__(self, tokenizer: BPETokenizer | None = None) -> None:
        self.tokenizer = tokenizer or BPETokenizer()
        self.signifier = NLPSignifier()
        self.domain = Domain()
        self.word_bits: dict[str, int] = {}
        self.pending: list[Pending] = []
        self.stores: list[tuple[int, Kline]] = []  # (step, store)
        self.cogitator = Cogitator(self.domain)  # the default runner; swap in ReferenceCogitator for the baseline

    def run(self, source: str, resolver=None) -> list[Step]:
        """An ask==goal run: every `==` ask with a resolvable goal is one
        scenario, everything else primes the domain before rationalisation.
        Scenarios run in ask order over the one live domain — an earlier
        scenario's stores and proposals persist into the later ones."""
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
        for other in entries:
            if id(other) not in paired:
                self.domain.add(_to_kline(other))  # primed — never an R1 store
        steps: list[Step] = []
        for ask, goal in pairs:
            self._step(steps, len(steps), goal)  # the goal rationalises first — S1 grounds on receipt
            self._step(steps, len(steps), ask, goal=_to_kline(goal))  # the ask asks — the goal is its first candidate
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

    def _step(self, steps: list[Step], i: int, value: KValue, goal: Kline | None = None) -> None:
        """Present one entry: rationalise, record, discharge, file proposals."""
        kline = _to_kline(value)
        result = self.cogitator.rationalise(kline, goal=goal)
        steps.append(Step(i, value, kline, result))
        self._discharge(i, kline)  # only proposals from earlier steps answer
        for prop in proposals(result):
            self.pending.append(Pending(prop, i))
        for store in result.stores:
            self.stores.append((i, store))

    def _discharge(self, step_index: int, entry: Kline) -> None:
        """A presented entry overlapping a pending target discharges it —
        proposals from earlier steps only (the raiser never answers itself)."""
        content = entry.sig | can(entry.adj)
        for p in self.pending:
            if p.raised_at < step_index and p.discharged_at is None and content & p.prop.kline.sig:
                p.discharged_at = step_index


def _to_kline(value: KValue) -> Kline:
    """The compiled entry as a content-space kline (appendix: cargo masked,
    labels preserved — nodes stay KNode throughout)."""
    sig = value.kline.signature
    masked_sig = KNode(int(sig) & WORD_MASK, getattr(sig, "label", ""))
    adj = frozenset(
        KNode(int(n) & WORD_MASK, getattr(n, "label", "")) for n in value.kline.nodes
    )
    return Kline(masked_sig, adj)


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


def _render(k: Kline, labels: dict[int, str], alloc: dict[int, str], verbose: bool) -> str:
    nodes = ", ".join(_label(n, labels, alloc, verbose) for n in sorted(k.adj))
    return f"{_label(k.sig, labels, alloc, verbose)}:[{nodes}]"


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


def _initials(parts: list[str]) -> str:
    """The Uppercase Character notation (MTS): a single signature stays
    whole; a compound of two or more takes each part whole when it is
    already an abbreviation, else its initial."""
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return "".join(p if p.isupper() else p[:1].upper() for p in parts)


def _tlabel(v: int, labels: dict[int, str], alloc: dict[int, str]) -> str:
    """A table label: a held label or word decomposition in MTS form —
    MarySubject -> MS, did|have -> DH, had a little lamb -> HALL."""
    if not v:
        return ""
    name = labels.get(int(v)) or getattr(v, "label", "")
    if name and "|" in name:
        return _initials([p for p in name.split("|") if p])
    if name and "∖" not in name and "∧" not in name and name != "⊥":
        return _initials(_humps(name))
    return _initials(_words(v, labels, alloc))


def _tframe(v: int, labels: dict[int, str], alloc: dict[int, str]) -> str:
    """Frame/event content in MTS form."""
    return _tlabel(v, labels, alloc)


_EVENT_KIND = {"close": "resolved", "wait": "stalled", "known": "known",
               "stall": "stalled", "exchange": "exchange", "pair": ""}


def _event_line(ev: Event, labels: dict[int, str], alloc: dict[int, str]) -> list[str]:
    c = ""
    if ev.candidate is not None:
        c = _tlabel(ev.candidate.sig, labels, alloc)
        if ev.candidate.adj and can(ev.candidate.adj) != ev.candidate.sig:
            c += ":" + " ".join(_tlabel(n, labels, alloc) for n in sorted(ev.candidate.adj))
    kind = "grounded" if ev.kind == "close" and ev.depth == 1 else _EVENT_KIND.get(ev.kind, ev.kind)
    return [
        str(ev.depth),
        _tlabel(ev.entry, labels, alloc),
        c,
        _tlabel(ev.F, labels, alloc),
        _tlabel(ev.U, labels, alloc),
        _tlabel(ev.O, labels, alloc),
        _tframe(int(ev.frame), labels, alloc) if ev.frame else "",
        kind,
    ]


def _render_events(events: list[Event], labels: dict[int, str], alloc: dict[int, str]) -> list[str]:
    """The event table: depth, entry, candidate, fit, underfit, overfit,
    frame, event — one line per event, in order."""
    hdr = ["depth", "E(ntry)", "C(andidate)", "F(it)", "U(nderfit)", "O(verfit)", "frame", "event"]
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


def _tline(k: Kline, labels: dict[int, str], alloc: dict[int, str]) -> str:
    """A kline in table form — sig in MTS form, nodes spelled out."""
    sig = _tlabel(k.sig, labels, alloc)
    nodes = ", ".join(_tlabel(n, labels, alloc) for n in sorted(k.adj))
    return f"{sig}:[{nodes}]" if k.adj else f"{sig}:[]"


def _render_step(step: Step, labels: dict[int, str], alloc: dict[int, str], verbose: bool) -> str:
    lines = [
        f"── Step {step.index + 1}  {band(step.kline).value}  "
        f"{_render(step.kline, labels, alloc, verbose)} ──"
    ]
    if step.result.events:
        if step.result.claim is None:
            lines.append("      (no claim — every event below)")
        lines.extend("      " + l for l in _render_events(step.result.events, labels, alloc))
    for store in step.result.stores:
        lines.append(f"      stores    {_tline(store, labels, alloc)}")
    by_reason: dict[str, list[End]] = {}
    for e in step.result.ends:
        by_reason.setdefault(e.reason, []).append(e)
    for reason, ends in by_reason.items():
        if all(e.kline is None for e in ends):  # an outcome, not a halt
            for _ in ends:
                lines.append(f"      end       {reason}")
            continue
        exemplars = [e for e in ends if e.kline is not None][:2]
        for e in exemplars:
            lines.append(
                f"      halt      {reason} — {_tline(e.kline, labels, alloc)}"
                f"  E={e.effort}"
            )
        rest = len(ends) - len(exemplars)
        if rest > 0:
            lines.append(f"      halt      {reason} ×{len(ends)} (+{rest} more)")
    return "\n".join(lines)


def present(
    steps: list[Step],
    domain: Domain,
    pending: list[Pending],
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
        print(_render_step(step, labels, alloc, verbose))
    if not verbose:
        return
    print("\n── domain (held at end of run) ──")
    for k in sorted(domain, key=lambda x: (x.sig.bit_length(), x.sig)):
        mark = "canon " if can(k.adj) == k.sig else "claim "
        print(f"      {mark}{_render(k, labels, alloc, verbose)}")
    print("\n── pending proposals (open at end of run) ──")
    open_props = [p for p in pending if p.discharged_at is None]
    open_props.sort(key=lambda p: p.prop.effort)  # most significant first (Definition 15)
    for p in open_props[:8]:
        print(
            f"      {p.prop.kind:<8} raised at step {p.raised_at + 1}  "
            f"{_render(p.prop.kline, labels, alloc, verbose)}  E={p.prop.effort}"
        )
    if len(open_props) > 8:
        print(f"      … +{len(open_props) - 8} more")
    if not open_props:
        print("      (none)")
    discharged = [p for p in pending if p.discharged_at is not None]
    if discharged:
        print(
            f"\n── discharged: {len(discharged)} proposals answered by later entries ──"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a .ks script through the rationalisation model.")
    parser.add_argument("source", help="Path to a .ks KScript file")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show the end-of-run state (held domain, open proposals) and hex "
             "signatures alongside labels.",
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
        steps = runner.run(source, resolver=path_resolver(path.parent))
    except ValueError as exc:
        print(f"harness: {exc}", file=sys.stderr)
        return 2
    present(
        steps, runner.domain, runner.pending, source, runner.tokenizer,
        runner.signifier, verbose=args.verbose, word_bits=runner.word_bits,
        resolver=path_resolver(path.parent),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
