---
type: source
title: "Observation: TRS spec implemented and tested: one spec defect fixed (emptied expansion)"
tags:
  - trs
  - implementation
  - testing
  - spec-defect
  - python
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-trs-spec-implemented-and-tested-one-spec-defect-fixed-emptie
relevance: high
observed_at: 2026-10-01T11:45:37.745Z
source_context: Testing docs/trs-significance.md through code (src/trs implementation)
---

# ⭐ Observation: TRS spec implemented and tested: one spec defect fixed (emptied expansion)

Spec tested through code: src/trs/ package (model.py, ~200 lines) + tests/test_trs_model.py (19 tests, all passing; full suite 126 passing, no engine regressions). Implementation maps 1:1 to spec sections: Node algebra (word-bit), Kline (validates non-empty head per Definition 2), Domain with canonical index + witness (total: ∅ for unheld) + three lookups + i1_closed() checker, band (ordered Def 11 cases), decompose (F/U/O), r2_products (three klines, non-empty heads, cross-pairing), r3_substitute (single pass, occurrence licence, original-adjacency occurrence test), rationalise runner (FIFO + seen-dedupe + max_rounds bound — all §7 strategy). FINDINGS from testing: (1) SPEC DEFECT fixed: R3's product ⟨can(N′),N′⟩ is ill-formed when substitution empties the adjacency (can(∅)=⊥ not a legal head per Definition 2) — spec amended (round 10): 'an expansion that empties N′ arrives at R4'; R4 arrival restated as the single condition 'no adjacency' with three forms. Runner implements this. (2) Behavioural fact: the fit product ⟨F, witness(F)⟩ re-entry is largely idempotent — its witness's own kline yields an S1 candidate that is already held, so re-entry mints nothing; a mint through re-entry requires a DIFFERENT non-canonical neighbour whose adjacency composes the same content (test: domain k(AB,{A,B}) canonical + k(ABC,{AB}) non-canonical; entry ⟨C,{C}⟩ → S3 → R3 → ⟨AB,{A,B}⟩ re-enters → S1 mints k(AB,{AB})). R1 mints exactly when a candidate is a re-canonicalisation of a held misfit adjacency. (3) Pure-overfit reentry cycles confirmed empirically (fit product re-derives the same S2) — §7's 'arrival is strategy' is load-bearing; runner's dedupe+bound handles it. (4) Presentation-time I-1 not enforced by the runner (caller contract); i1_closed() available for checking. Uncommitted at time of observation.

*Relevance: high*
*Context: Testing docs/trs-significance.md through code (src/trs implementation)*
*Tags: trs implementation testing spec-defect python*

---
*Observed: 2026-10-01T11:45:37.745Z*
