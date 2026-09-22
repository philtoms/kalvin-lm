---
type: source
title: "Observation: cogitate single-pass; callers own the reentry loop"
tags:
  - refactor
  - cogitator
  - reentry
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-cogitate-single-pass-callers-own-the-reentry-loop
relevance: medium
observed_at: 2026-09-21T17:34:42.925Z
source_context: Single-pass cogitate refactor
---

# 🔍 Observation: cogitate single-pass; callers own the reentry loop

Cogitate is now single-pass: the internal recursion (`if count != len(work_list): batch.extend(cogitate(state))`) is gone from kalvin.cogitator — `cogitate(state)` does exactly one oldest-first pass and returns. Reentry is the caller's job: each caller loops `while True: size = len(state.work_list); batch.extend(cogitate(state)); if len(state.work_list) == size: break` — re-entering while a pass changes the work-list length (semantically identical to the old recursion, byte-identical mhall.ks output). Callers: harness._drive turn, turn() helpers in probe_ask_fate/block_filter/wdmh_s3_recip, inline loops in probe_hop_wdmh/rationalise/wdmh_decode. Gotcha hit: the harness already used `before` for its _grounded_snapshot — the loop variable must not shadow it (renamed to `size`). This was stacked on the using_resolver move to dev callers and the per-probe turn() factoring, uncommitted together.

*Relevance: medium*
*Context: Single-pass cogitate refactor*
*Tags: refactor cogitator reentry*

---
*Observed: 2026-09-21T17:34:42.925Z*
