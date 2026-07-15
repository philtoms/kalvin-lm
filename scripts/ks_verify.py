#!/usr/bin/env python3
"""Compile a KScript source and verify each generated kline.

Accepts a filepath (to a .ks file) or a raw source string, runs the
full compilation pipeline (Lexer → Parser → AST Emitter → Token
Encoder), and prints a structured verification of every kline grouped
by semantic section.

Usage:
    python scripts/ks_verify.py
    python scripts/ks_verify.py path/to/script.ks
    python scripts/ks_verify.py "A == B"
    python scripts/ks_verify.py script.ks --tokenizer kalvin
    python scripts/ks_verify.py script.ks --raw
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalvin.kline import KLine, sig_level
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source

_signifier = NLPSignifier()

# Default curriculum script — Mary Had A Little Lamb (MHALL).
# Exercises every operator: COUNTERSIGNS, CANONIZES, DENOTES, CONNOTES,
# and the corpus annotation + subscript scaffolding forms.
DEFAULT_SOURCE = """\
(Mary had a little lamb)
MHALL == SVO =>
   S(ubject) = M
   V(erb) = H
   O(bject) = ALL =>
     A > D(et)
     L > M(od)
     L > O
"""


# ── Display helpers ───────────────────────────────────────────────────


def dec(tokenizer, n: int) -> str:
    """Decode a uint64 token to a readable string, fallback to hex."""
    try:
        r = tokenizer.decode([n])
        if r:
            return r
    except Exception:
        pass
    return f"#{n:#x}"


def dec_nodes(tokenizer, kl: KLine) -> list[str]:
    """Decode all node tokens in a kline."""
    return [dec(tokenizer, n) for n in kl.nodes]


def fmt_kline(tokenizer, kl: KLine, idx: int, width: int = 3) -> str:
    """Format a single kline as a one-line verification row."""
    level = sig_level(kl, _signifier)
    op = kl.dbg.op
    label = kl.dbg.label
    decoded_sig = dec(tokenizer, kl.signature)

    if kl.nodes:
        nodes = dec_nodes(tokenizer, kl)
        return f"  [{idx:{width}d}] {level:3s} {op:12s} | {label} [{decoded_sig!r}] => {nodes}"
    return f"  [{idx:{width}d}] {level:3s} {op:12s} | {label} [{decoded_sig!r}]"


def print_section(title: str, tokenizer, entries: list[KLine], start: int, end: int) -> None:
    """Print a labelled section of klines."""
    if start >= end:
        return
    w = len(str(len(entries) - 1))
    print(f"\n--- {title} [{start}–{end - 1}] ---")
    for i in range(start, end):
        print(fmt_kline(tokenizer, entries[i], i, width=w))


# ── Section detection ─────────────────────────────────────────────────
#
# The klines follow a regular structure produced by the compiler pipeline.
# We detect section boundaries by tracking operator transitions and
# label changes, then group them into named sections.


def _detect_sections(entries: list[KLine]) -> list[tuple[str, int, int]]:
    """Detect semantic sections in a kline list.

    The compiler emits compiled source before any MTS entries: operator
    klines (COUNTERSIGNS/DENOTES/CONNOTES) come first, followed by MTS
    expansion klines (IDENTITY components and CANONIZES aggregates — both
    §8 character-level and §11.3 BPE-subword). We surface this as two macro
    sections in that order. A source that emits bare identity klines with
    no operator (e.g. §14.8) is folded into the MTS section by this
    op-only heuristic; each row still carries its own op label.
    """
    if not entries:
        return []

    def is_source(kl: KLine) -> bool:
        return (kl.dbg.op if kl.dbg else "") in (
            "COUNTERSIGNS",
            "DENOTES",
            "CONNOTES",
        )

    # Source occupies a leading contiguous run; the remainder is MTS.
    boundary = 0
    while boundary < len(entries) and is_source(entries[boundary]):
        boundary += 1

    sections: list[tuple[str, int, int]] = []
    if boundary > 0:
        sections.append(("Compiled source (operators)", 0, boundary))
    if boundary < len(entries):
        sections.append((
            "MTS expansion (identities & canonizations)",
            boundary,
            len(entries),
        ))
    return sections


# ── Summary ───────────────────────────────────────────────────────────


def print_summary(entries: list[KLine]) -> None:
    """Print operator counts and significance level distribution."""
    from collections import Counter

    op_counts = Counter(kl.dbg.op for kl in entries if kl.dbg)
    level_counts = Counter(sig_level(kl, _signifier) for kl in entries)

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Total klines: {len(entries)}")
    print()

    print(f"  {'Op':15s} {'Count':>6s}")
    print(f"  {'-' * 22}")
    for op, count in sorted(op_counts.items()):
        print(f"  {op:15s} {count:6d}")

    print()
    print(f"  {'Level':6s} {'Count':>6s}")
    print(f"  {'-' * 14}")
    for level in ("S1", "S2", "S3", "S4"):
        if level in level_counts:
            print(f"  {level:6s} {level_counts[level]:6d}")


# ── Tokenizer loading ────────────────────────────────────────────────


def load_tokenizer(name: str):
    """Load a tokenizer by name.

    The kalvin tokenizer is the sole production tokenizer; ``name`` must
    be ``"kalvin"``.
    """
    if name == "kalvin":
        from kalvin.nlp_tokenizer import NLPTokenizer

        return NLPTokenizer(), "kalvin"
    else:
        raise ValueError(f"Unknown tokenizer: {name}")


# ── Main ──────────────────────────────────────────────────────────────


def run(source: str, tokenizer_name: str, raw: bool = False) -> None:
    """Compile source and print verification."""
    tokenizer, tok_label = load_tokenizer(tokenizer_name)

    compiled = compile_source(source, tokenizer, dev=True)
    # compile_source returns KValue (kline + significance); the display path
    # works on the underlying KLine, so unwrap once.
    entries: list[KLine] = [kv.kline for kv in compiled]

    w = len(str(len(entries) - 1)) if entries else 1

    print("=" * 78)
    print(f"KLINE VERIFICATION  ({tok_label} tokenizer)")
    print("=" * 78)

    if raw:
        # Raw mode: just print every kline in order
        print()
        for i, kl in enumerate(entries):
            print(fmt_kline(tokenizer, kl, i, width=w))
    else:
        # Structured mode: detect and print sections
        sections = _detect_sections(entries)
        for title, start, end in sections:
            print_section(title, tokenizer, entries, start, end)

    print_summary(entries)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile KScript and verify each kline",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help=(
            "KScript source string, or path to a .ks file. "
            "If omitted, defaults to the Mary Had A Little Lamb "
            "(MHALL) reference script."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        choices=["kalvin"],
        default="kalvin",
        help="Tokenizer to use (default: kalvin — the sole production tokenizer).",
    )
    parser.add_argument(
        "--raw",
        "-r",
        action="store_true",
        help="Print raw sequential klines without section grouping",
    )
    args = parser.parse_args()

    # Resolve source: default, file path, or literal string
    source_arg = args.source if args.source is not None else DEFAULT_SOURCE
    source_path = Path(source_arg)
    if source_path.exists() and source_path.is_file():
        source = source_path.read_text(encoding="utf-8")
        label = source_path.name
    else:
        source = source_arg
        label = "<default>" if args.source is None else "<string>"

    print(f"Source: {label}")

    run(source, args.tokenizer, raw=args.raw)


if __name__ == "__main__":
    main()
