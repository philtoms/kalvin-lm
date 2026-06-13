"""Quick smoke test for the Kalvin agent pipeline.

Compiles the MHALL-SVO curriculum, feeds it to KAgent, and prints
a summary of rationalisation events. Uses EventBus (no harness)
for lightweight testing.

Usage:
    python scripts/kalvin_test.py          # summary mode (default)
    python scripts/kalvin_test.py --verbose # show every event
"""

import argparse
import sys
import threading
from collections import Counter

sys.path.insert(0, "src")

from kalvin.agent import KAgent
from kalvin.events import EventBus, RationaliseEvent
from kalvin.expand import D_MAX, boundaries, classify
from kalvin.nlp_tokenizer import NLPTokenizer
from ks.compiler import compile_source

SOURCE = """
(Mary had a little lamb)
MHALL == SVO =>
   S(ubject) = M
   V(erb) = H
   O(bject) = ALL =>
     A > D(et)
     L > M(od)
     L > O
"""

# ── Helpers ───────────────────────────────────────────────────────────


def significance_level(sig: int) -> str:
    """Classify a raw significance value into S1–S4."""
    if sig == D_MAX - 1:
        return "S1"
    if sig == 0:
        return "S4"
    s12, s23, s34 = boundaries()
    return classify(sig, s12, s23, s34)


def kline_display(kline, _tokenizer) -> str:
    """Human-readable display for a KLine."""
    if kline is None:
        return "<none>"
    # Use rich dbg if available
    if kline.dbg:
        dbg = kline.dbg
        label = dbg.label
        # Show decoded text if it differs from the label
        if dbg.decoded and dbg.decoded != label:
            label = f"{label} [{dbg.decoded!r}]"
        # Append NLP info
        nlp_parts = []
        if dbg.pos:
            nlp_parts.append(dbg.pos)
        if dbg.dep:
            nlp_parts.append(dbg.dep)
        if nlp_parts:
            label += f" ({'/'.join(nlp_parts)})"
        nodes = kline.nodes
        if not nodes:
            return label
        node_strs = [_tokenizer.decode([n]) or f"#{n:#x}" for n in nodes]
        return f"{label}: {node_strs}"
    # Fallback: raw sig
    return f"sig={kline.signature:#x}"


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalvin agent smoke test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show every event")
    args = parser.parse_args()

    tokenizer = NLPTokenizer.from_files()
    klines = compile_source(SOURCE, tokenizer, dev=True)

    adapter = EventBus()
    agent = KAgent(tokenizer=tokenizer, adapter=adapter)

    done_event = threading.Event()
    counts: Counter = Counter()
    s1_entries: list[str] = []

    def on_event(e: RationaliseEvent) -> None:
        if e.kind == "done":
            done_event.set()
            return

        level = significance_level(e.significance)
        key = f"{e.kind}:{level}"
        counts[key] += 1

        if args.verbose:
            query_str = kline_display(e.query, tokenizer)
            proposal_str = kline_display(e.proposal, tokenizer)
            arrow = "←" if e.significance == D_MAX - 1 else "|"
            print(f"  {e.kind:6s} {query_str} → {level} {arrow} {proposal_str}")

        if level == "S1" and e.kind != "ground":
            query_str = kline_display(e.query, tokenizer)
            proposal_str = kline_display(e.proposal, tokenizer)
            s1_entries.append(f"{query_str} ← {proposal_str}")

    adapter.subscribe(on_event)

    # Print compiled entries
    print("Compiled entries:")
    for k in klines:
        print(f"  {kline_display(k, tokenizer)}")

    # Rationalise
    print("\nRationalising...")
    for k in klines:
        agent.rationalise(k)

    done_event.wait()
    agent.cogitate_join()

    # Print summary
    print(f"\nEvent summary ({sum(counts.values())} total):")
    for key in sorted(counts):
        print(f"  {key:12s} {counts[key]:>4d}")

    if s1_entries:
        print(f"\nS1 frames ({len(s1_entries)}):")
        for entry in s1_entries:
            print(f"  {entry}")

    print("\nDone.")


if __name__ == "__main__":
    main()
