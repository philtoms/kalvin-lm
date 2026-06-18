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


def _decode_value(value: int, model, tokenizer) -> str:
    """Decode a single signature/node value to text via the graph.

    Tries each kline headed by *value* (most-recent-first) and unpacks the
    first that decomposes to identity tokens. This skips semantic-
    relationship klines (connoted/undersigned) that share a signature with
    a canonize, landing on the structural decomposition. Falls back to a
    direct decode for single tokens, then to a hex label — never raises.
    """
    if model is not None:
        for kl in model.klines():
            if kl.signature != value:
                continue
            try:
                text = tokenizer.decode(model.unpack(kl))
                if text:
                    return text
            except ValueError:
                continue
    try:
        text = tokenizer.decode([value])
        if text:
            return text
    except Exception:
        pass
    return f"#{value:#x}"


def _nlp_suffix(dbg) -> str:
    """Build the POS/dep suffix from a dbg record, if any."""
    if dbg is None:
        return ""
    nlp_parts = []
    if dbg.pos:
        nlp_parts.append(dbg.pos)
    if dbg.dep:
        nlp_parts.append(dbg.dep)
    return f" ({'/'.join(nlp_parts)})" if nlp_parts else ""


def kline_display(kline, tokenizer, model=None) -> str:
    """Human-readable display for a KLine.

    When *model* is supplied, signature and node values are decoded by
    flattening each through model.unpack (correct for packed/multi-token
    values); the dbg record, if present, contributes only NLP metadata.
    Otherwise (no model) the dbg label and a direct per-node decode are
    used (lossy for packed values, but the best available before the
    graph is populated). Falls back to a raw signature if neither is
    available.
    """
    if kline is None:
        return "<none>"

    dbg = getattr(kline, "dbg", None)

    # Primary path: graph-based decode.
    if model is not None:
        decoded = _decode_value(kline.signature, model, tokenizer)
        label = decoded or (dbg.label if dbg else f"#{kline.signature:#x}")
        label += _nlp_suffix(dbg)
        nodes = kline.nodes
        if not nodes:
            return label
        node_strs = [_decode_value(n, model, tokenizer) or f"#{n:#x}" for n in nodes]
        return f"{label}: {node_strs}"

    # Fallback path: compile-time dbg (lossy for packed values).
    if dbg:
        label = dbg.label
        if dbg.decoded and dbg.decoded != label:
            label = f"{label} [{dbg.decoded!r}]"
        label += _nlp_suffix(dbg)
        nodes = kline.nodes
        if not nodes:
            return label
        node_strs = [_decode_value(n, None, tokenizer) or f"#{n:#x}" for n in nodes]
        return f"{label}: {node_strs}"

    # Last resort: raw signature.
    return f"sig={kline.signature:#x}"


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalvin agent smoke test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show every event")
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=5.0,
        help="Seconds to wait for the 'done' event before giving up (default: 5).",
    )
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
            query_str = kline_display(e.query, tokenizer, agent.model)
            proposal_str = kline_display(e.proposal, tokenizer, agent.model)
            arrow = "←" if e.significance == D_MAX - 1 else "|"
            print(f"  {e.kind:6s} {query_str} → {level} {arrow} {proposal_str}")

        if level == "S1" and e.kind != "ground":
            query_str = kline_display(e.query, tokenizer, agent.model)
            proposal_str = kline_display(e.proposal, tokenizer, agent.model)
            s1_entries.append(f"{query_str} ← {proposal_str}")

    adapter.subscribe(on_event)

    if args.verbose:
        # Build a display model from the compiled klines so that packed
        # signatures can be unpacked to their identity tokens for display
        # (the agent model is still empty at this point).
        from kalvin.model import Model

        display_model = Model()
        for k in klines:
            display_model.add_to_frame(k)

        # Print compiled entries
        print("Compiled entries:")
        for k in klines:
            print(f"  {kline_display(k, tokenizer, display_model)}")

    # Rationalise
    print("\nRationalising...")
    for k in klines:
        agent.rationalise(k)

    if not done_event.wait(timeout=args.timeout):
        print(
            f"\nWARNING: no 'done' event after {args.timeout:.1f}s — "
            "continuing anyway."
        )
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
