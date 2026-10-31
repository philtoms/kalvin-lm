---
type: source
title: "Observation: Runner consolidation: rationalise3 is now rationalise; nested-end propagation fixed"
tags:
  - trs
  - refactor
  - reentry
status: observation
created: 2026-10-12
updated: 2026-10-12
slug: obs-2026-10-12-runner-consolidation-rationalise3-is-now-rationalise-nested-
relevance: high
observed_at: 2026-10-12T16:20:05.998Z
source_context: Removing rationalise/rationalise2, renaming rationalise3 to rationalise in src/trs
---

# ⭐ Observation: Runner consolidation: rationalise3 is now rationalise; nested-end propagation fixed

rationalise (LIFO stack runner) and rationalise2 (step-1.2 prototype) deleted; rationalise3 renamed to rationalise (src/trs/model.py). r2_products (cross-wise witness-lookup variant, dead after runner removal) also deleted; r2_products3 renamed to r2_products. Callers updated: trs/__init__.py exports, dev/dialogue/trs_harness.py. 10 old-runner test pins removed/adapted (opener-minting, LIFO starvation, max_rounds bounds, witness R2 products). Latent bug found and fixed during the rename: rationalise3 discarded nested Rationalisation results, so nested halts vanished — a top-level entry can never produce a bound end (fresh seen set, depth 0), making Definition 15's continue proposal unreachable and starving the dialogue harness of proposals. Fix: S2/S3 branches now extend the caller's ends with nested ends. Spec touched up: D15 continue bullet now names the halted branch (repeated canon / depth) instead of the LIFO minimal-effort pending kline; §7 drops "round length".

*Relevance: high*
*Context: Removing rationalise/rationalise2, renaming rationalise3 to rationalise in src/trs*
*Tags: trs refactor reentry*

---
*Observed: 2026-10-12T16:20:05.998Z*
