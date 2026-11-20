---
type: source
title: "Observation: KLine migration value-parity verified against HEAD"
tags:
  - rationalisation
  - parity
  - verification
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-kline-migration-value-parity-verified-against-head
relevance: high
observed_at: 2026-11-17T09:00:15.503Z
source_context: KLine type migration verification
---

# ⭐ Observation: KLine migration value-parity verified against HEAD

Migration verified behaviour-preserving: the value-space of every harness step (claims, stores, ends, event F/U/O/frame values) hashes identical to HEAD for both canonical scripts (mhall.ks, wdmh.ks). The only observable differences are display-level: node listings render in authored order (was set-iteration order ≈ sorted), and derived-region labels compose in authored order (e.g. WDMH's O-region shows HLAL where HEAD showed HALL) because KNode.merge preserves encounter order and the OR-fold is commutative on values. The 2 test failures on the suite (test_no_fabrication, test_rationalisation_is_honest) are pre-existing on HEAD: goal-less rationalise(d, entry) crashes at entry.cands=[None] in the Cogitator — not a regression. Parked instruments (verify.py, existence.py) still carry pre-existing pyflakes 'undefined name side' debt.

*Relevance: high*
*Context: KLine type migration verification*
*Tags: rationalisation parity verification*

---
*Observed: 2026-11-17T09:00:15.503Z*
