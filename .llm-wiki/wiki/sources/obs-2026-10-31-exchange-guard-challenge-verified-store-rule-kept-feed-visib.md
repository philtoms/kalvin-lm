---
type: source
title: "Observation: Exchange guard challenge verified: store-rule kept, feed-visibility decoupled"
tags:
  - trs
  - exchange
  - guard
  - verification
  - bundling
  - feed-visibility
status: observation
created: 2026-10-31
updated: 2026-10-31
slug: obs-2026-10-31-exchange-guard-challenge-verified-store-rule-kept-feed-visib
relevance: high
observed_at: 2026-10-31T08:27:38.671Z
source_context: Verifying user challenge to the _exchange composition guard
---

# ⭐ Observation: Exchange guard challenge verified: store-rule kept, feed-visibility decoupled

User challenged the _exchange composition guard (docstring: "The mutual witness stores only where it composes content beyond the signature — a self-witnessing pair (Mod:[Mod]) acknowledges, it does not learn"), asserting the two live runs are unaffected when the code is commented out. Verification (A/B diff of full harness output): HALF RIGHT. mhall: byte-identical (no exchange feeds its path — zero admits). wdmh: grounding unaffected, but the commented state lost the entire S3 chain (23→12 tabled rows) and the DH:[did,have] store — the guard ALSO gated realx, which drove exchange-feed tableding. The real defect exposed: BUNDLING — one condition gating two concerns (the R1 store vs trace visibility). Fix (commit b9cf855): the composition rule gates ONLY the admit (the docstring sentence is now exactly what the code does); feed visibility derives from exchange causality on the walked path — _trace marks a stall's parent row as a feed when tabling the exchange pair, exempt from the slot claim but never claiming it; realx and sticky sets deleted. Both scripts byte-identical to the guarded baseline; 176 passed. Residual: on live runs no vacuous exchange ever admits (instrumented: mhall 0 admits, wdmh only DH:[did,have]), so the guard's store-half is currently unobservable — it stands as R1 semantics, one if, awaiting a script that exercises it.

*Relevance: high*
*Context: Verifying user challenge to the _exchange composition guard*
*Tags: trs exchange guard verification bundling feed-visibility*

---
*Observed: 2026-10-31T08:27:38.671Z*
