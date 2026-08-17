---
type: source
title: "Observation: K-driven harness rewrite: mhall completes, 22 groundings"
tags:
  - dialogue
  - harness
  - engine
status: observation
created: 2026-08-17
updated: 2026-08-17
slug: obs-2026-08-17-k-driven-harness-rewrite-mhall-completes-22-groundings
relevance: high
observed_at: 2026-08-17T14:54:41.696Z
source_context: Lean harness rewrite in src/dialogue/harness.py + engine cascade fix
---

# ⭐ Observation: K-driven harness rewrite: mhall completes, 22 groundings

Rewrote Harness.run as a K-driven dialogue: each sub-script (annotation group) opens with its first entry; from there every engine emission is an ask, answered from the script or the run stops. Ratifying answers: identity ask X:[] → X:[X] + script klines headed X; proposal ask A:[B] → matching script kline + countersignature B:[A]. Compound self-ref entries (DH:[did,have], WDMH:[what,did,Mary,have]) mark their nodes as script-known words. mhall now runs to completion: 5 steps, no stops, STM empty, 22 groundings including MHALL:[Mary,had,a,little,lamb], SVO, ALL, and WDMH:[what,did,Mary,have]. Earlier in the session also fixed an Engine._ground cascade bug (it grounded `kline` instead of the groundable STM `entry`). Open frontier: no relationship reaches the S3 countersign arm — the misfit arm pops entries even when propose() returns nothing; candidate fix is popping only on a non-empty batch.

*Relevance: high*
*Context: Lean harness rewrite in src/dialogue/harness.py + engine cascade fix*
*Tags: dialogue harness engine*

---
*Observed: 2026-08-17T14:54:41.696Z*
