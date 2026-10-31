---
type: source
title: "Observation: TRS spec: S4 at reentry via empty adjacency; R3 single outcome; R4 single arrival"
tags:
  - trs
  - significance
  - s4
  - reentry
  - rules
  - outcomes
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-trs-spec-s4-at-reentry-via-empty-adjacency-r3-single-outcome
relevance: high
observed_at: 2026-10-01T10:32:02.060Z
source_context: Moving S4 transition to reentry; single-outcome R3; single-arrival R4
---

# ⭐ Observation: TRS spec: S4 at reentry via empty adjacency; R3 single outcome; R4 single arrival

Tenth round on docs/trs-significance.md, three interlocking rulings: (1) R2 no longer states the unheld-residual S4 transition — it's now a reentry concern. Mechanism: a residual not held has no decode; its product carries the EMPTY adjacency (stated in R2 as a decode note), and the §5 reentry paragraph carries the concern: 'A product with the empty adjacency re-enters as an entry kline with no adjacency and routes to R4 — the misfit asks, and later-arriving information may complete the path.' Note the asymmetry this creates: an unheld HEAD residual is harmless (an entry's signature is a lookup key), only the CARRIED residual's decode failure yields the empty adjacency. Also 'three klines — for the non-empty residuals' (⊥ heads skipped in pure-direction S2). (2) R3 has a single outcome: the reentrant kline ⟨can(N′), N′⟩ — the no-substitutions → R4 outcome deleted (N′=N simply yields ⟨can(N), N⟩). (3) R4 has a single arrival: an entry kline with no adjacency (the two-arrival list collapsed to one condition — a product whose residual had no adjacency to carry, or a candidate whose neighbour supplied none per Def 11 case 1). D8/D11 rows and the diagram updated; amendment note at six rounds. Definition 6's tail repointed from R2 to reentry. Rules now have clean uniform shape: R1 stores, R2 → three reentrant klines, R3 → one reentrant kline, R4 ends on no-adjacency arrival; all S4 semantics live in the entry pipeline + R4 arrival condition.

*Relevance: high*
*Context: Moving S4 transition to reentry; single-outcome R3; single-arrival R4*
*Tags: trs significance s4 reentry rules outcomes*

---
*Observed: 2026-10-01T10:32:02.060Z*
