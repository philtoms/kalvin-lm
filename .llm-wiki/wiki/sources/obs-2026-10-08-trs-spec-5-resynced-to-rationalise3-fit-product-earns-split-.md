---
type: source
title: "Observation: TRS spec §5 resynced to rationalise3: fit product earns, split returns on first cross S1"
tags:
  - trs
  - significance
  - reentry
  - r2-earn
  - r1-licensed-claim
status: observation
created: 2026-10-08
updated: 2026-10-08
slug: obs-2026-10-08-trs-spec-5-resynced-to-rationalise3-fit-product-earns-split-
relevance: high
observed_at: 2026-10-08T22:08:16.815Z
source_context: Bringing docs/trs-significance.md to spec with rationalise3
---

# ⭐ Observation: TRS spec §5 resynced to rationalise3: fit product earns, split returns on first cross S1

Twelfth round on docs/trs-significance.md: §5 brought up to spec with rationalise3 (src/trs/model.py, the standardised reentry run by dev/dialogue/trs_harness.py). Key algebra changes: (1) R2's fit product does NOT re-enter — its adjacency is earned unconditionally into the running composition that joins later candidates' adjacencies; only the cross products re-enter, in order, and the FIRST whose rationalisation returns at S1 returns the split (replacing "at least two products return at S1"). (2) R1 stores a licensed claim, not necessarily a canon — "claim and content coincide" restricted to direct S1. (3) No early termination: every candidate is processed; each returning candidate stores and CLEARS the accumulator. (4) D13's R2 entropy bullet was internally inconsistent with R2 (still witness-lookup phrasing "H=0 when its head is held"); rewritten to reconstruction semantics: cross products carry H=1 always (claims disjoint from â by construction), fit product H=0 node-clean (can(N_F)=F), and Σμ(head)·H(product) ≥ μ(σ∨â)·H(C) with equality iff the F-side split is node-clean. (5) §7 bounds extended with "repeated entry canons" (rationalise3's seen-memo). (6) R4 enumeration now includes "a presented entry". Code repairs alongside: restored the R4 guard for no-adjacency presented entries (rationalise3 crashed with ValueError — regression from the restructure); removed a dead duplicate rationalise3 call in the S2 branch (no-op due to seen-memo); fixed stale probe import (r2_products3 not exported from trs). Suite 159 green.

*Relevance: high*
*Context: Bringing docs/trs-significance.md to spec with rationalise3*
*Tags: trs significance reentry r2-earn r1-licensed-claim*

---
*Observed: 2026-10-08T22:08:16.815Z*
