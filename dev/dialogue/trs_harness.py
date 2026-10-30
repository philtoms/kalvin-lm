"""The TRS dialogue path — the same scripts, the significance-routed model.

Compiles a KScript source exactly as the engine path does, then runs it
as an ask==goal configuration: the script's last `==` ask and its goal
kline are the two entries — the ask first, the goal answers — and every
other kline primes the domain before rationalisation. Priming is a
harness fact, never a model admission: R1 remains the sole way the run
itself learns. Nothing is checked at presentation; unheld references
find no neighbours.

Proposals persist across steps: a later presented entry discharges a
pending proposal when its content overlaps the target (Definition 15).
The presenter reports; it never judges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.kline import KNode, is_ask
from kalvin.kvalue import KValue
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source, path_resolver

from trs import (
    Domain,
    End,
    Kline,
    Proposal,
    Rationalisation,
    Row,
    Cogitator,
    band,
    can,
    proposals,
)

from dev.dialogue.harness import _sig_to_label

# The appendix's content mask: the word half is content; the token half is
# inert encoding cargo, masked off at every content read. The TRS model
# operates in content space — values are masked once, here at the boundary.
WORD_MASK = 0x7FFF_FFFF_0000_0000


@dataclass
class TrsStep:
    """One presented entry and what its rationalisation produced."""

    index: int
    entry: KValue  # the compiled entry (annotation/label live here)
    kline: Kline  # the TRS form
    result: Rationalisation


@dataclass
class Pending:
    """A proposal awaiting an answer, and its fate."""

    prop: Proposal
    raised_at: int
    discharged_at: int | None = None


class TrsRunner:
    """Compile → present → rationalise → record, over one live Domain."""

    def __init__(self, tokenizer: BPETokenizer | None = None) -> None:
        self.tokenizer = tokenizer or BPETokenizer()
        self.signifier = NLPSignifier()
        self.domain = Domain()
        self.word_bits: dict[str, int] = {}
        self.pending: list[Pending] = []
        self.stores: list[tuple[int, Kline]] = []  # (step, store)
        self.cogitator = Cogitator(self.domain)
    def run(self, source: str, resolver=None) -> list[TrsStep]:
        """An ask==goal run: the `==` ask and its goal are the entries,
        everything else primes the domain before rationalisation."""
        entries = compile_source(
            source,
            tokenizer=self.tokenizer,
            signifier=self.signifier,
            dev=True,
            word_bits=self.word_bits,
            resolver=resolver,
        )
        pair = self._ask_goal_pair(entries)
        if pair is None:
            raise ValueError("ask==goal: no `==` ask with a findable goal kline")
        ask, goal = pair
        for other in entries:
            if other is not ask and other is not goal:
                self.domain.add(_to_kline(other))  # primed — never an R1 store
        steps: list[TrsStep] = []
        self._step(steps, 0, goal)  # the goal rationalises first — S1 grounds on receipt
        self._step(steps, 1, ask, goal=_to_kline(goal))  # the ask asks — the goal is its first candidate
        return steps

    @staticmethod
    def _ask_goal_pair(entries: list[KValue]) -> tuple[KValue, KValue] | None:
        """The script's last `==` ask whose named goal resolves to a kline
        entry — the final question is the point of the script. The goal is
        the last label match (the fullest authored definition): has nodes,
        not itself an ask."""
        pair: tuple[KValue, KValue] | None = None
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
                pair = (e, goal)
        return pair

    def _step(self, steps: list[TrsStep], i: int, value: KValue, goal: Kline | None = None) -> None:
        """Present one entry: rationalise, record, discharge, file proposals."""
        kline = _to_kline(value)
        result = self.cogitator.rationalise(kline, goal=goal)
        steps.append(TrsStep(i, value, kline, result))
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
    """Frame/step parts in MTS form."""
    return _tlabel(v, labels, alloc)


def _on_path_children(row: Row, keep: set[int]) -> list:
    """The children whose reentries carried the resolution — a resolving
    row, an exchange stall, or any kept row of the child is the continuation."""
    out = []
    for ch in row.children:
        if (
            (ch.resolved_by is not None and id(ch.resolved_by) in keep)
            or (ch.stalled is not None and id(ch.stalled) in keep)
            or any(id(r) in keep for r in ch.rows)
        ):
            out.append(ch)
    return out


def _row_step(row: Row, labels: dict[int, str], alloc: dict[int, str], keep: set[int]) -> str:
    if row.candidate is None:
        e = _tlabel(row.entry.sig, labels, alloc)
        return f"resolved {e}" if row.exchange else f"stalled {e}"
    children = [ch for ch in row.children if ch.popped]
    if children:
        return "reenter " + ", ".join(_tlabel(ch.kline.sig, labels, alloc) for ch in children)
    if row.resolved:
        parts = _tframe(row.F, labels, alloc)
        return f"resolved {parts}" if parts else "resolved"
    return ""


def _table_row(row: Row, labels: dict[int, str], alloc: dict[int, str], keep: set[int]) -> list[str]:
    e = _tlabel(row.entry.sig, labels, alloc)
    if row.candidate is not None:
        c = _tlabel(row.candidate.sig, labels, alloc)
        if row.candidate.adj and can(row.candidate.adj) != row.candidate.sig:
            c += ":" + " ".join(
                _tlabel(n, labels, alloc) for n in sorted(row.candidate.adj)
            )
    else:
        c = ""
    frame = _tframe(row.frame, labels, alloc)
    return [
        str(row.depth), e, c,
        _tlabel(row.F, labels, alloc),
        _tlabel(row.U, labels, alloc),
        _tlabel(row.O, labels, alloc),
        frame,
        _row_step(row, labels, alloc, keep),
    ]


def _tline(k: Kline, labels: dict[int, str], alloc: dict[int, str]) -> str:
    """A kline in table form — sig in MTS form, nodes spelled out."""
    sig = _tlabel(k.sig, labels, alloc)
    nodes = ", ".join(_tlabel(n, labels, alloc) for n in sorted(k.adj))
    return f"{sig}:[{nodes}]" if k.adj else f"{sig}:[]"


def _render_table(rows: list[Row], labels: dict[int, str], alloc: dict[int, str]) -> list[str]:
    """The trace table: depth, entry, candidate, fit, underfit, overfit,
    frame, step."""
    hdr = ["depth", "E(ntry)", "C(andidate)", "F(it)", "U(nderfit)", "O(verfit)", "frame", "step"]
    keep = {id(r) for r in rows}
    body = [_table_row(r, labels, alloc, keep) for r in rows]
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


def _render_step(step: TrsStep, labels: dict[int, str], alloc: dict[int, str], verbose: bool) -> str:
    lines = [
        f"── Step {step.index + 1}  {band(step.kline).value}  "
        f"{_render(step.kline, labels, alloc, verbose)} ──"
    ]
    rows = step.result.path if step.result.claim is not None else step.result.rows
    if rows:
        if step.result.claim is None:
            lines.append("      (no path — the goal was not reached; all rows below)")
        lines.extend("      " + l for l in _render_table(rows, labels, alloc))
    for store in step.result.stores:
        lines.append(f"      stores    {_tline(store, labels, alloc)}")
    by_reason: dict[str, list[End]] = {}
    for e in step.result.ends:
        by_reason.setdefault(e.reason, []).append(e)
    for reason, ends in by_reason.items():
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


def present_trs(
    steps: list[TrsStep],
    domain: Domain,
    pending: list[Pending],
    source: str,
    tokenizer: BPETokenizer,
    signifier: NLPSignifier,
    *,
    verbose: bool,
    word_bits: dict[str, int] | None = None,
    resolver=None,
) -> None:
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
    import argparse

    parser = argparse.ArgumentParser(description="Run a .ks script through the TRS model.")
    parser.add_argument("source", help="Path to a .ks KScript file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.source)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"trs-harness: could not read {args.source!r}: {exc}")
        return 2
    runner = TrsRunner()
    try:
        steps = runner.run(source, resolver=path_resolver(path.parent))
    except ValueError as exc:
        print(f"trs-harness: {exc}")
        return 2
    present_trs(
        steps, runner.domain, runner.pending, source, runner.tokenizer,
        runner.signifier, verbose=args.verbose, word_bits=runner.word_bits,
        resolver=path_resolver(path.parent),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
