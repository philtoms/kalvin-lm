---
type: source
title: One-directional coverage fractions peak at containment, not equality
status: insight
category: design
created: 2026-09-10
updated: 2026-09-10
slug: coverage-fractions-peak-at-containment
---

# One-directional coverage fractions peak at containment, not equality

When defining a continuous "distance/coverage" measure graded between discrete structural classes, a one-directional coverage fraction silently violates the top class: |σ_A ∧ σ_B|/|σ_A| = 1 whenever A's content is *contained* in B's — which in Kalvin's taxonomy is an Underfit/Overfit relationship (S2), not the exact class (S1, value-equality). Any measure meant to be maximal exactly at equality must weigh both directions: what A's content covers of B *and* what B's covers of A — the Jaccard form |A ∧ B|/|A ∨ B| is the canonical depth-free core (0 exactly at disjointness, 1 exactly at equality). Kalvin's per-slot accountedness α(n) = |n ∧ σ(ν_B)|/|n| composes across slots and carries a hop-discount for witness depth (making σ-preserving granularity moves strictly decrease the measure — the detector for gratuitous expansion). General lesson: for each "the measure is maximal exactly when X" requirement, construct the near-miss case (containment without equality) and check the proposed formula against it before accepting it. Context: [[sources/obs-2026-09-10-layers-3-4-review-ks2-10-12-selection-clauses-split-band-or]].

*Category: design*

---
*Captured: 2026-09-10*

## Related

_Add links to related pages._
