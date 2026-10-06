"""The TRS dialogue path — the same scripts, the significance-routed model.

Compiles a KScript source exactly as the engine path does, then presents
each entry to ``trs.rationalise`` in authored order — one step per entry,
D live throughout (D14). Word identities are seeded as the presentation
substrate (tokenizer-derivable facts, as the engine path's batch primes);
R1 remains the sole admission for everything the run itself learns.

Proposals persist across steps: a later presented entry discharges a
pending proposal when its content overlaps the target (Definition 15).
The presenter reports; it never judges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.kline import KNode
from kalvin.kvalue import KValue
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source, path_resolver

from trs import (
    Domain,
    End,
    Kline,
    Proposal,
    Rationalisation,
    band,
    can,
    proposals,
    rationalise,
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

    def run(self, source: str, resolver=None) -> list[TrsStep]:
        entries = compile_source(
            source,
            tokenizer=self.tokenizer,
            signifier=self.signifier,
            dev=True,
            word_bits=self.word_bits,
            resolver=resolver,
        )
        steps: list[TrsStep] = []
        for i, value in enumerate(entries):
            kline = _to_kline(value)
            result = rationalise(self.domain, kline)
            steps.append(TrsStep(i, value, kline, result))
            self._discharge(i, kline)  # only proposals from earlier steps answer
            for prop in proposals(result):
                self.pending.append(Pending(prop, i))
            for store in result.stores:
                self.stores.append((i, store))
        return steps

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


def _render_step(step: TrsStep, labels: dict[int, str], alloc: dict[int, str], verbose: bool) -> str:
    lines = [
        f"── Step {step.index + 1}  {band(step.kline).value}  "
        f"{_render(step.kline, labels, alloc, verbose)} ──"
    ]
    for store in step.result.stores:
        lines.append(f"      stores    {_render(store, labels, alloc, verbose)}")
    by_reason: dict[str, list[End]] = {}
    for e in step.result.ends:
        by_reason.setdefault(e.reason, []).append(e)
    for reason, ends in by_reason.items():
        exemplars = [e for e in ends if e.kline is not None][:2]
        for e in exemplars:
            lines.append(
                f"      halt      {reason} — {_render(e.kline, labels, alloc, verbose)}"
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
    open_props.sort(key=lambda p: p.prop.effort)  # most significant first (D13)
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
    steps = runner.run(source, resolver=path_resolver(path.parent))
    present_trs(
        steps, runner.domain, runner.pending, source, runner.tokenizer,
        runner.signifier, verbose=args.verbose, word_bits=runner.word_bits,
        resolver=path_resolver(path.parent),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
