"""sig8.demo — walk through the grill's defining scenarios, printed.

Run: ``python -m prototype.sig8.demo``

Not a test — a narrative. Shows the compose-on-return aggregation producing
bytes for the cases the grill kept returning to:
  - exact match           -> 0xFF
  - total non-account      -> 0x00
  - 3-node vs 30-node S2   -> same byte (count-invariance, Q10)
  - deeper resolution      -> lower byte (Q13)
  - swapping the seams     -> different bytes (Q16)
"""

from __future__ import annotations

from prototype.sig8.aggregate import Aggregator
from prototype.sig8.byte import BandLayout
from prototype.sig8.functions import (
    make_asymptotic_decay,
    mean_compose,
    min_compose,
)


def _show(label: str, r) -> None:
    layout = BandLayout()
    band = layout.classify(r.significance)
    print(
        f"  {label:<42} "
        f"frac={r.accounted_fraction:.4f}  byte={r.significance:#04x}  band={band}"
    )


def main() -> None:
    agg = Aggregator()  # default seams: asymptotic k=50 + mean compose

    print("=== saturation guards (Q9) ===")
    _show("exact match [1.0, 1.0]", agg.compose_terminal([1.0, 1.0]))
    _show("total non-account [0.0, 0.0]", agg.compose_terminal([0.0, 0.0]))
    _show("interior [1.0, 0.0]", agg.compose_terminal([1.0, 0.0]))

    print("\n=== count-invariance (Q10) ===")
    _show("3-node  [1.0, 0.5, 0.0]", agg.compose_terminal([1.0, 0.5, 0.0]))
    _show("30-node [1.0, 0.5, 0.0]*10", agg.compose_terminal([1.0, 0.5, 0.0] * 10))

    print("\n=== decay curves (Q12) — deeper resolution -> lower byte ===")
    for h in (1, 2, 5, 20, 100):
        _show(f"single slot, {h} hops", agg.compose_terminal([agg.decay(h)]))

    print("\n=== swapping decay (Q16, seam 1) at 10 hops ===")
    slow = Aggregator(compose=mean_compose, decay=make_asymptotic_decay(k=100))
    fast = Aggregator(compose=mean_compose, decay=make_asymptotic_decay(k=5))
    _show("k=100 (slow decay)", slow.compose_terminal([slow.decay(10)]))
    _show("k=5   (fast decay)", fast.compose_terminal([fast.decay(10)]))

    print("\n=== swapping compose (Q16, seam 2) on [1.0, 0.2] ===")
    mean_agg = Aggregator(compose=mean_compose)
    min_agg = Aggregator(compose=min_compose)
    _show("mean_compose", mean_agg.compose_terminal([1.0, 0.2]))
    _show("min_compose", min_agg.compose_terminal([1.0, 0.2]))

    print("\n=== band boundary knob (Q4/Q5) — same byte, different band ===")
    byte = agg.compose_terminal([1.0, 0.5]).significance
    for b in (0x20, 0x40, 0x80, 0xC0):
        layout = BandLayout(s2_s3_boundary=b)
        print(f"  byte={byte:#04x}  S2_S3_BOUNDARY={b:#04x}  -> band={layout.classify(byte)}")


if __name__ == "__main__":
    main()
