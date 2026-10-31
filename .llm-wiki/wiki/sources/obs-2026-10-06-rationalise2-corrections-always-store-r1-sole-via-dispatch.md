---
type: source
title: "Observation: rationalise2 corrections: always-store, R1 sole via dispatch"
tags:
  - trs
  - redesign
  - rationalise2
  - r1
  - store
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-rationalise2-corrections-always-store-r1-sole-via-dispatch
relevance: medium
observed_at: 2026-10-06T16:34:18.542Z
source_context: Correcting rationalise2 store semantics
---

# 🔍 Observation: rationalise2 corrections: always-store, R1 sole via dispatch

TWO CORRECTIONS to rationalise2 (user rulings): (1) store must be TRULY ALWAYS — the R1 arm now does `d.add(cand); stores.append(cand)` unconditionally; every store event is recorded including duplicates (d.add stays idempotent for D; the log never filters); (2) R1 is the SOLE store point via dispatch — the store() closure removed; dispatch's S1 arm is the only writing code, and the enriched-F path no longer special-cases S1: it iterates back through dispatch(enriched), so an S1 enriched kline stores through the same R1 arm. mhall goal trace unchanged (3 stores: Object, VO, SVO; 0 ends; no duplicates arise on this script). 159 tests passing.

*Relevance: medium*
*Context: Correcting rationalise2 store semantics*
*Tags: trs redesign rationalise2 r1 store*

---
*Observed: 2026-10-06T16:34:18.542Z*
