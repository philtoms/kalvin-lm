"""Quick smoke test for the Kalvin agent pipeline.

Compiles the MHALL-SVO curriculum, feeds it to KAgent, and prints
a summary of rationalisation events. Uses EventBus (no harness)
for lightweight testing.

Usage:
    python scripts/kalvin_test.py          # summary mode (default)
    python scripts/kalvin_test.py --verbose # show every event
"""

import argparse
import signal
import sys
import threading
from collections import Counter

sys.path.insert(0, "src")

from kalvin.agent import KAgent
from kalvin.events import EventBus, RationaliseEvent
from kalvin.expand import D_MAX, SIG_S4, boundaries, classify
from kalvin.kline import is_identity
from kalvin.kvalue import KValue
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

(what did Mary have)
WDMH =>
  M
  H => D H
  W = ALL
"""

class DeadlineExceededError(Exception):
    """Raised when a wall-clock --deadline elapses mid-run."""


# ── Helpers ──────────────────────────────────────────────────────────


def significance_level(sig: int) -> str:
    """Classify a raw significance value into S1–S4."""
    if sig == D_MAX:
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
    hex label — never raises.
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

    return f"#{value:#x}"


def _type_suffix(dbg) -> str:
    """Build the type-info suffix from a dbg record, if any."""
    if dbg is None:
        return ""
    return f" ({dbg.type_info})" if dbg.type_info else ""


def kvalue_display(kvalue, tokenizer, model=None) -> str:
    """Human-readable display for a KValue, prefixed with its significance band.

    The KValue's significance is converted to an ``S1``–``S4`` label and
    prefixed (e.g. ``[S2] Mary: ['M', 'ary']``), so each displayed kline
    carries its own assessment — the two-voice view (query = sender's
    declared significance, proposal = Kalvin's) is visible at a glance.

    When *model* is supplied, signature and node values are decoded by
    flattening each through model.unpack (correct for packed/multi-token
    values); the dbg record, if present, contributes only type metadata.
    Otherwise (no model) the dbg label and a direct per-node decode are
    used (lossy for packed values, but the best available before the
    graph is populated). Falls back to a raw signature if neither is
    available.
    """
    if kvalue is None:
        return "<none>"

    level = significance_level(kvalue.significance)
    kline = kvalue.kline
    dbg = getattr(kline, "dbg", None)

    body: str
    # Primary path: graph-based decode.
    if model is not None:
        decoded = _decode_value(kline.signature, model, tokenizer)
        label = decoded or (dbg.label if dbg else f"#{kline.signature:#x}")
        label += _type_suffix(dbg)
        nodes = kline.nodes
        if not nodes:
            body = label
        else:
            node_strs = [_decode_value(n, model, tokenizer) or f"#{n:#x}" for n in nodes]
            body = f"{label}: {node_strs}"
    # Fallback path: compile-time dbg (lossy for packed values).
    elif dbg:
        label = dbg.label
        if dbg.decoded and dbg.decoded != label:
            label = f"{label} [{dbg.decoded!r}]"
        label += _type_suffix(dbg)
        nodes = kline.nodes
        if not nodes:
            body = label
        else:
            node_strs = [_decode_value(n, None, tokenizer) or f"#{n:#x}" for n in nodes]
            body = f"{label}: {node_strs}"
    # Last resort: raw signature.
    else:
        body = f"sig={kline.signature:#x}"

    return f"[{level}] {body}"


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
    parser.add_argument(
        "--deadline",
        type=float,
        default=10.0,
        help=(
            "Wall-clock limit in seconds for the WHOLE run. If the"
            " rationalise/cogitate phase exceeds it, the script prints the"
            " partial S1 count gathered so far and exits 124 (timeout)."
            " 0 = disabled. Use this to bound runs against the"
            " unfixed Model/STM race, which can livelock this script."
        ),
    )
    args = parser.parse_args()

    tokenizer = NLPTokenizer()
    kvalues = compile_source(SOURCE, tokenizer, dev=True)

    adapter = EventBus()
    agent = KAgent(tokenizer=tokenizer, adapter=adapter)

    done_event = threading.Event()
    counts: Counter = Counter()
    entries: dict[str,list[str]] = {
        "S1":[],
        "S2":[],
        "S3":[],
        "S4":[],
    }

    def on_event(e: RationaliseEvent) -> None:
        if e.kind == "done":
            done_event.set()
            return

        # Events carry two KValues (KE-3): query (sender's declared
        # assessment) and proposal (Kalvin's assessment). Significance lives
        # on the KValue, not the event.
        level = significance_level(e.proposal.significance)
        counts[level] += 1

        if args.verbose:
            query_str = kvalue_display(e.query, tokenizer, agent.model)
            proposal_str = kvalue_display(e.proposal, tokenizer, agent.model)
            # The arrow marks S1 ratification (←) vs not (|); each side's
            # significance band is already shown by its [Sx] prefix.
            arrow = "←" if e.proposal.significance == D_MAX else "|"
            print(f"  {e.kind:6s} {query_str} {arrow} {proposal_str}")

        # if e.kind != "ground":
        query_str = kvalue_display(e.query, tokenizer, agent.model)
        proposal_str = kvalue_display(e.proposal, tokenizer, agent.model)
        entries[level].append(f"{query_str} ← {proposal_str}")

    adapter.subscribe(on_event)

    if args.verbose:
        # Build a display model from the compiled klines so that packed
        # signatures can be unpacked to their identity tokens for display
        # (the agent model is still empty at this point).
        from kalvin.model import Model

        display_model = Model()
        for k in kvalues:
            display_model.add_to_frame(k.kline)

        # Print compiled entries (each compiled entry is a KValue — KP-1).
        print("Compiled entries:")
        for k in kvalues:
            print(f"  {kvalue_display(k, tokenizer, display_model)}")

    # Rationalise
    print("\nRationalising...")

    # Install a wall-clock deadline so an unfixed Model/STM race cannot
    # livelock this script forever. SIGALRM only works on the main thread,
    # which is exactly where this runs.
    prev_handler = None
    if args.deadline > 0:

        def _alarm_handler(signum, frame):
            raise DeadlineExceededError()

        prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(round(args.deadline)))

    deadline_hit = False
    try:
        # agent.rationalise(kvalues[0])
        for k in kvalues:
            agent.rationalise(k)

        if not done_event.wait(timeout=args.timeout):
            print(f"\nWARNING: no 'done' event after {args.timeout:.1f}s.")
        agent.cogitate_join()
    except DeadlineExceededError:
        deadline_hit = True
    finally:
        if args.deadline > 0:
            signal.alarm(0)  # cancel any pending alarm
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)

    if deadline_hit:
        print(
            f"\nDEADLINE_EXCEEDED: --deadline {args.deadline:.1f}s elapsed "
            "mid-run (likely the unfixed Model/STM race livelocking the "
            "rationalise/cogitate phase). Reporting partial counts."
        )

    # Print summary
    print(f"\nEvent summary ({sum(counts.values())} total):")
    for key in sorted(counts):
        print(f"  {key:12s} {counts[key]:>4d}")

    for level in ["S1", "S2", "S3","S4"]:
        if entries[level]:
            print(f"\n{level} frames ({len(entries[level])}):")
            for entry in entries[level][:30]:
                print(f"  {entry}")

    print("\nDone.")
    if deadline_hit:
        # 124 is the conventional exit code for a timed-out command
        # (matches GNU `timeout`), so callers/gates can detect it.
        sys.exit(124)


if __name__ == "__main__":
    main()
