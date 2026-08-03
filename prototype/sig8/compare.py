"""sig8.compare — run the same scenarios as tests/test_expand.py, before/after.

Prints a table: for each scenario, the production significance (sum-and-invert)
vs the prototype significance (compose-on-return byte), plus the band each
lands in. Lets us *observe* whether the redesign behaves as expected.

Run: ``python -m prototype.sig8.compare``
"""

from __future__ import annotations

from kalvin.expand import D_MAX, MASK64, MAX_HOP, S2_S3_DISTANCE, boundaries, classify, expand
from kalvin.kline import KLine
from kalvin.model import Model
from kalvin.signifier import NLPSignifier

from .byte import BandLayout
from .aggregate import Aggregator
from .expand_proto import expand_proto

signifier = NLPSignifier()


def t(bits: int) -> int:
    """Place sig-word bits in the upper 32 bits (mirrors test_expand)."""
    return bits << 32


def _prod_summary(model, q, c):
    results = list(expand(model, q, c, signifier))
    s12, s23, s34 = boundaries()
    terminal = results[-1]
    n_side = len(results) - 1
    return terminal.significance, classify(terminal.significance, s12, s23, s34), n_side


def _proto_summary(model, q, c, agg, layout):
    results = list(expand_proto(model, q, c, signifier, agg))
    # The top-level pair's terminal is the LAST yield (mirrors production's
    # results[-1] convention); nested recursive calls yield their own
    # terminals earlier in the stream.
    terminal = results[-1]
    assert terminal.kind == "terminal"
    n_side = len(results) - 1
    return terminal.significance, layout.classify(terminal.significance), n_side, terminal.result


def _row(name, prod_sig, prod_band, proto_sig, proto_band, n_side_prod, n_side_proto, frac):
    print(
        f"  {name:<46} "
        f"prod={prod_sig & MASK64 & 0xFFFFFFFFFFFFFFFF if prod_sig else 0}".replace("prod=0", "prod=    0")  # noqa
    )
    # (the f-string above is messy; use a cleaner format below instead)


def main() -> None:
    layout = BandLayout()  # default boundary 0x80
    agg = Aggregator(layout=layout)
    s12, s23, s34 = boundaries()

    print("=" * 100)
    print("sig8 before/after — production (sum-and-invert) vs prototype (compose-on-return)")
    print(f"prototype layout: S2_S3_BOUNDARY={layout.s2_s3_boundary:#04x}  "
          f"decay=asymptotic(k=50)  compose=mean")
    print("=" * 100)
    print(f"  {'scenario':<42} {'prod sig':>14} {'band':>4}   "
          f"{'proto byte':>10} {'band':>4} {'frac':>6}   {'side yields':>11}")
    print("  " + "-" * 96)

    def scenario(name, model, q, c):
        prod_sig, prod_band, n_side_prod = _prod_summary(model, q, c)
        proto_sig, proto_band, n_side_proto, term = _proto_summary(model, q, c, agg, layout)
        prod_disp = f"{prod_sig & MASK64:#x}"
        proto_hex = f"{proto_sig:#04x}"
        print(
            f"  {name:<42} {prod_disp:>14} {prod_band:>4}   "
            f"{proto_hex:>10} {proto_band:>4} {term.accounted_fraction:>6.3f}   "
            f"{n_side_prod:>2} -> {n_side_proto:<2}"
        )

    # ── Scenarios lifted from tests/test_expand.py ────────────────────

    # test_expand_self_no_model: all matched, none grounded -> 3 ungrounded matches.
    m = Model(); k = KLine(10, [10, 20, 30])
    scenario("self, 3 matched-ungrounded nodes", m, k, k)

    # test_expand_no_resolution: 1 matched, 4 unresolvable mismatches.
    m = Model(); q = KLine(5, [1, 2, 3]); c = KLine(6, [1, 4, 5])
    scenario("1 matched + 4 unresolvable", m, q, c)

    # test_expand_with_grounding: 1 grounded match, 2 unresolvable.
    m = Model(); m.add_to_frame(KLine(0b110, [0b100, 0b010]))
    q = KLine(5, [0b110, 2]); c = KLine(6, [0b110, 3])
    scenario("1 grounded + 2 unresolvable", m, q, c)

    # test_expand_all_matched_grounded: 2 grounded matches -> prod D_MAX.
    m = Model()
    m.add_to_frame(KLine(0b110, [0b100, 0b010]))
    m.add_to_frame(KLine(0b1100, [0b1000, 0b0100]))
    q = KLine(5, [0b110, 0b1100]); c = KLine(6, [0b110, 0b1100])
    scenario("2 grounded matches (prod = D_MAX)", m, q, c)

    # test_expand_hop_reaches_opposing_mismatch: 1 hop resolves + unresolvable.
    m = Model()
    m.add_to_frame(KLine(t(0b110), [t(0b100), t(0b010)]))
    m.add_to_frame(KLine(t(20), [t(0b110)]))
    m.add_to_frame(KLine(t(10), [t(20)]))
    m.add_to_frame(KLine(t(5), [t(10)]))
    q = KLine(100, [t(5), t(2)]); c = KLine(200, [t(10), t(3)])
    scenario("1 hop-resolves + unresolvable (+side)", m, q, c)

    # test_expand_signifies_cogitation: a signifies side-candidate is yielded.
    m = Model()
    m.add_to_frame(KLine(t(30), [t(30)]))
    m.add_to_frame(KLine(t(20), [t(30)]))
    m.add_to_frame(KLine(t(10), [t(20)]))
    m.add_to_frame(KLine(t(5), [t(10)]))
    q = KLine(100, [t(5)]); c = KLine(200, [t(10)])
    scenario("signifies side-candidate (+side)", m, q, c)

    # test_expand_connotation_bridging: an S3 connotation bridge.
    m = Model()
    m.add_to_frame(KLine(8, [8]))
    m.add_to_frame(KLine(4, [8]))
    m.add_to_frame(KLine(2, [8]))
    q = KLine(100, [4]); c = KLine(200, [2])
    scenario("S3 connotation bridge (+side)", m, q, c)

    # All matched, all ungrounded — many nodes (degradation sweep).
    m = Model(); k = KLine(10, list(range(30)))
    scenario("30 matched-ungrounded nodes", m, k, k)

    # ── Band sanity: does the prototype land intuitively? ─────────────
    print()
    print("prototype band legend:  S1=[0xFF]  S2=[0x80,0xFE]  S3=[0x01,0x7F]  S4=[0x00]")
    print("note: 'side yields' = N(signifies+connotation candidates) yielded besides terminal")


if __name__ == "__main__":
    main()
