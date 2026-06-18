"""KB-309 significance probe — exercises the REAL expand()/classify() path
(the only place UNRESOLVED_PENALTY has an effect) across the value sweep.

Two representative unresolved-misfit scenarios, mirroring the curriculum's
intent and the TestS2Gradient regression:

  A. Symmetric disjoint pairs (empty model): n mismatched nodes each side.
     Terminal distance = 2 * n * UNRESOLVED_PENALTY. (TestS2Gradient scenario.)

  B. Single-sided unresolved ("dual-misfit" analog {MH:[H,A]} vs {M:[H]}):
     1 matched node + k extra unmatched query nodes. Each extra contributes
     UNRESOLVED_PENALTY. Terminal distance = k * UNRESOLVED_PENALTY.

For each swept value we report the S2/S3/S4 band distribution: how many
unresolved-node counts stay in S2 before spilling into S3. A healthy value
keeps a visible, non-trivial S2 band distinct from S3.

Run: PYTHONPATH=src python auto-tune/penalty-sweep/probe_significance.py
"""

from __future__ import annotations

import kalvin.expand as ex
from kalvin.expand import MASK64, S2_S3_DISTANCE, boundaries, classify
from kalvin.kline import KLine
from kalvin.model import Model

SWEEP = [5, 8, 10, 12, 15, 20]


def band_for(model: Model, q: KLine, c: KLine) -> tuple[int, str]:
    terminal = list(ex.expand(model, q, c))[-1]
    dist = (~terminal.significance) & MASK64
    s12, s23, s34 = boundaries()
    return dist, classify(terminal.significance, s12, s23, s34)


def scenario_symmetric(v: int) -> dict:
    """n disjoint nodes each side → distance 2*n*v."""
    ex.UNRESOLVED_PENALTY = v
    m = Model()
    rows = []
    for n in range(1, 11):
        q = KLine(1, list(range(1, n + 1)))
        c = KLine(2, list(range(1000, 1000 + n)))
        dist, band = band_for(m, q, c)
        rows.append((n, dist, band))
    return {"rows": rows, "s2_capacity": sum(1 for _, _, b in rows if b == "S2")}


def scenario_dualmisfit(v: int) -> dict:
    """1 matched node + k extra query nodes → distance k*v (single-sided)."""
    ex.UNRESOLVED_PENALTY = v
    m = Model()
    rows = []
    for k in range(1, 11):
        # query: node 0 (matches) + k extra unmatched nodes
        q_nodes = [0] + list(range(100, 100 + k))
        c_nodes = [0]
        q = KLine(1, q_nodes)
        c = KLine(2, c_nodes)
        dist, band = band_for(m, q, c)
        rows.append((k, dist, band))
    return {"rows": rows, "s2_capacity": sum(1 for _, _, b in rows if b == "S2")}


def main() -> None:
    print(f"S2_S3_DISTANCE = {S2_S3_DISTANCE}  (S2 band = distance 1..{S2_S3_DISTANCE})")
    print()
    print("SCENARIO A — symmetric disjoint pairs (distance = 2*n*v):")
    print(f"{'value':>5} | {'S2-fit n':>8} | {'first S3 n':>10} | bands (n=1..10)")
    print("-" * 70)
    for v in SWEEP:
        res = scenario_symmetric(v)
        bands = " ".join(b for _, _, b in res["rows"])
        first_s3 = next((n for n, _, b in res["rows"] if b == "S3"), None)
        print(f"{v:>5} | {res['s2_capacity']:>8} | {str(first_s3):>10} | {bands}")
    print()
    print("SCENARIO B — dual-misfit analog (distance = k*v, single-sided):")
    print(f"{'value':>5} | {'S2-fit k':>8} | {'first S3 k':>10} | bands (k=1..10)")
    print("-" * 70)
    for v in SWEEP:
        res = scenario_dualmisfit(v)
        bands = " ".join(b for _, _, b in res["rows"])
        first_s3 = next((k for k, _, b in res["rows"] if b == "S3"), None)
        print(f"{v:>5} | {res['s2_capacity']:>8} | {str(first_s3):>10} | {bands}")
    print()
    print("Analytic S2 capacity (unresolved pairs before S3 spill):")
    for v in SWEEP:
        cap_sym = S2_S3_DISTANCE // (2 * v)  # symmetric: 2*v per pair
        cap_one = S2_S3_DISTANCE // v  # single-sided: v each
        print(f"  v={v:>2}: symmetric-pair S2 capacity = {cap_sym}; single-sided = {cap_one}")


if __name__ == "__main__":
    main()
