---
type: source
title: "Observation: Effort defined: E discounts repairable residual; S3 strictly below S4"
tags:
  - trs
  - effort
  - bias
  - repairability
  - definition-14
  - halt
  - proposals
  - beta
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-effort-defined-e-discounts-repairable-residual-s3-strictly-b
relevance: high
observed_at: 2026-10-01T13:36:00.308Z
source_context: Biasing entropy by repairability for halt-time proposals
---

# ⭐ Observation: Effort defined: E discounts repairable residual; S3 strictly below S4

EFFORT defined (amendment 13, Definition 14, docs/trs-significance.md): E(C) = (μ(U) + β·μ(O))/μ(σ ∨ â), β ∈ (0,1) named constant default ½ — entropy biased by repairability, motivated by the user's intuition 'S3 is strictly less surprising than S4 — there is repairability at S3'. KEY INSIGHT: no symmetric explained-fraction measure can split S3 from S4 (at S3 everything in play is unexplained, F = ⊥) — the split must credit O-side mass for being PRESENT and dereferenceable while U-side needs acquisition. Anchors hold: E(S1) = 0 (canon and identity alike), E(S4) = 1 exactly; E(S3) < 1 pointwise (β·μ(â) < μ(â) since â ≠ ⊥); S2 interior both kinds (β > 0 keeps pure overfit off zero). β → 1 recovers H; E ≤ H; R2 entropy theorem stated for H only, does NOT transfer. An S2 may outweigh an S3 — effort grades within bands, the band still routes. PURPOSE: 'Effort is the halt measure' — when rationalisation ends, E of the halted kline = how far the trace did not get = ranking fuel for PROPOSALS GENERATED AT HALT (the user's next planned spec improvement: proposals at the halt point, measured by residual entropy). Implementation: BETA = Fraction(1,2), effort(k, beta=BETA) in src/trs/model.py; End dataclass now carries effort: Fraction | None (the halt measure computed at each end with a kline); runner wires effort into all kline-bearing ends. D12 decision row added; amendment 13 logged. Tests: anchors, pointwise S3 < S4 (same claim C: k(C,{A,B}) = 2/3 vs k(C,∅) = 1), S2 interior both kinds, β-interpolation (effort(k, beta=1) == entropy(k)), ends carry the halt measure; property suite: effort anchors + S3-strict + S2-interior + E ≤ H + β-interpolation over random klines. 147 tests passing. Uncommitted (entropy work + effort).

*Relevance: high*
*Context: Biasing entropy by repairability for halt-time proposals*
*Tags: trs effort bias repairability definition-14 halt proposals beta*

---
*Observed: 2026-10-01T13:36:00.308Z*
