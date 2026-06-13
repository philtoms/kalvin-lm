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
    python scripts/ks_verify.py script.ks --tokenizer mod32
    python scripts/ks_verify.py script.ks --tokenizer nlp
    python scripts/ks_verify.py script.ks --tokenizer both
    python scripts/ks_verify.py script.ks --raw
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ks.compiler import compile_source
from kalvin.kline import KLine, KDbg, sig_level
from kalvin.nlp_tokenizer import NLPTokenizer


# Default curriculum script — Mary Had A Little Lamb (MHALL).
# Exercises every operator: COUNTERSIGN, CANONIZE, UNDERSIGN, CONNOTATE,
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
    level = sig_level(kl)
    op = kl.dbg.op
    label = kl.dbg.label
    decoded_sig = dec(tokenizer, kl.signature)

    if kl.nodes:
        nodes = dec_nodes(tokenizer, kl)
        return (
            f"  [{idx:{width}d}] {level:3s} {op:12s} | "
            f"{label} [{decoded_sig!r}] => {nodes}"
        )
    return (
        f"  [{idx:{width}d}] {level:3s} {op:12s} | "
        f"{label} [{decoded_sig!r}]"
    )


def print_section(title: str, tokenizer, entries: list[KLine],
                  start: int, end: int) -> None:
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

    Returns list of (title, start, end) tuples where end is exclusive.
    """
    if not entries:
        return []

    sections: list[tuple[str, int, int]] = []

    # --- Phase 1: Corpus word MCS blocks ---
    # A corpus word block is: N× IDENTITY (subwords), 1× CANONIZE, 1× IDENTITY (compound)
    # We detect the boundary where the label changes.

    i = 0
    corpus_start = i

    # Walk until we see the first non-corpus label (not a word from annotation)
    # Corpus labels appear as the first group of entries whose labels are
    # words from the opening annotation. They end when we encounter a label
    # that is a multi-char identifier NOT from the annotation.
    # Heuristic: corpus entries are IDENTITY+CANONIZE blocks where each
    # CANONIZE's nodes are all single-char or NLP subwords.

    # Find where corpus ends: first entry whose label differs from previous
    # AND is an identifier (uppercase letters, no lowercase word pattern).
    # Simpler: group consecutive entries that share the same label into blocks,
    # then classify blocks as corpus or structural.

    # Group by label runs
    groups: list[tuple[str, int, int]] = []  # (label, start, end)
    current_label = entries[0].dbg.label if entries[0].dbg else ""
    group_start = 0

    for j in range(len(entries)):
        kl = entries[j]
        label = kl.dbg.label if kl.dbg else ""
        if label != current_label:
            groups.append((current_label, group_start, j))
            current_label = label
            group_start = j
    groups.append((current_label, group_start, len(entries)))

    # Classify groups
    # Corpus words: groups whose label is lowercase (Mary, had, a, little, lamb, etc.)
    # Then structural: MHALL, S/V/O, SVO, label expansions (Subject, Verb, Object, etc.)
    # Then operators: COUNTERSIGN, UNDERSIGN, CONNOTATE

    def is_corpus_label(label: str) -> bool:
        """Corpus words start with an uppercase letter but aren't all-caps identifiers."""
        if not label:
            return False
        # All-caps multi-char = identifier (MHALL, SVO, ALL)
        if len(label) > 1 and label.isupper():
            return False
        # Single char = structural (S, V, O, A, L)
        if len(label) == 1:
            return False
        return True

    def is_operator(op: str) -> bool:
        return op in ("COUNTERSIGN", "UNDERSIGN", "CONNOTATE")

    # Build sections from groups
    idx = 0

    # Collect corpus groups
    corpus_groups: list[tuple[str, int, int]] = []
    while idx < len(groups) and is_corpus_label(groups[idx][0]):
        corpus_groups.append(groups[idx])
        idx += 1

    # Emit corpus sections
    for label, gs, ge in corpus_groups:
        word = label
        sections.append((f'Corpus: "{word}"', gs, ge))

    if not sections:
        # No corpus detected — just emit everything as one section
        return [("KLines", 0, len(entries))]

    # Collect structural groups (identifiers, label expansions) until operators
    structural_start = None
    operator_groups: list[tuple[str, int, int]] = []

    while idx < len(groups):
        label, gs, ge = groups[idx]
        # Check if this group contains any operator entries
        has_operator = any(
            is_operator(entries[j].dbg.op)
            for j in range(gs, ge)
            if entries[j].dbg
        )
        if has_operator:
            break
        if structural_start is None:
            structural_start = gs
        structural_end = ge
        idx += 1

    if structural_start is not None:
        sections.append(("Structural: identifiers & labels",
                         structural_start, structural_end))

    # Remaining groups: split into operator sections by type
    remaining_start = sections[-1][2] if sections else 0

    # Fine-grained: walk entries from remaining_start, group consecutive
    # entries by their operator type into sub-sections
    j = remaining_start
    while j < len(entries):
        kl = entries[j]
        op = kl.dbg.op if kl.dbg else "IDENTITY"

        if op == "COUNTERSIGN":
            cs_start = j
            while j < len(entries) and entries[j].dbg and entries[j].dbg.op == "COUNTERSIGN":
                j += 1
            sections.append(("COUNTERSIGN", cs_start, j))
        elif op == "UNDERSIGN":
            us_start = j
            while j < len(entries) and entries[j].dbg and entries[j].dbg.op == "UNDERSIGN":
                j += 1
            sections.append(("UNDERSIGN", us_start, j))
        elif op == "CONNOTATE":
            co_start = j
            while j < len(entries) and entries[j].dbg and entries[j].dbg.op == "CONNOTATE":
                j += 1
            sections.append(("CONNOTATE", co_start, j))
        else:
            # Structural or identity — batch until next operator
            st_start = j
            while j < len(entries):
                e_op = entries[j].dbg.op if entries[j].dbg else "IDENTITY"
                if e_op in ("COUNTERSIGN", "UNDERSIGN", "CONNOTATE"):
                    break
                j += 1
            sections.append(("Identifiers & labels", st_start, j))

    return sections


# ── Summary ───────────────────────────────────────────────────────────


def print_summary(entries: list[KLine]) -> None:
    """Print operator counts and significance level distribution."""
    from collections import Counter

    op_counts = Counter(kl.dbg.op for kl in entries if kl.dbg)
    level_counts = Counter(sig_level(kl) for kl in entries)

    print(f"\n{'=' * 78}")
    print(f"SUMMARY")
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
    """Load a tokenizer by name."""
    if name == "nlp":
        from kalvin.nlp_tokenizer import NLPTokenizer
        return NLPTokenizer.from_files(), "NLP"
    elif name == "mod32":
        from kalvin.mod_tokenizer import Mod32Tokenizer
        return Mod32Tokenizer(), "Mod32"
    else:
        raise ValueError(f"Unknown tokenizer: {name}")


# ── Main ──────────────────────────────────────────────────────────────


def run(source: str, tokenizer_name: str, raw: bool = False) -> None:
    """Compile source and print verification."""
    tokenizer, tok_label = load_tokenizer(tokenizer_name)

    entries = compile_source(source, tokenizer, dev=True)

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
        "--tokenizer", "-t",
        choices=["nlp", "mod32", "both"],
        default="nlp",
        help="Tokenizer to use (default: nlp). 'both' runs each and compares.",
    )
    parser.add_argument(
        "--raw", "-r",
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

    if args.tokenizer == "both":
        run(source, "nlp", raw=args.raw)
        run(source, "mod32", raw=args.raw)
    else:
        run(source, args.tokenizer, raw=args.raw)


if __name__ == "__main__":
    main()
