---
type: source
title: "Observation: WorkRunner cogitates the popped item, not the whole work list"
tags:
  - architecture
  - work-runner
  - cogitator
  - fix
status: observation
created: 2026-09-22
updated: 2026-09-22
slug: obs-2026-09-22-workrunner-cogitates-the-popped-item-not-the-whole-work-list
relevance: high
observed_at: 2026-09-22T08:48:50.902Z
source_context: Per-item cogitation in WorkRunner
---

# ⭐ Observation: WorkRunner cogitates the popped item, not the whole work list

Fixed the Engine/WorkRunner architecture flaw: the runner popped backlog items but cogitated the ENTIRE work list per item (the item was already in it — pops were mere pass tokens). cogitate(state, kline) now cogitates the one submitted kline (ground cascade / answer / propose / release, identity-based remove_work); cogitate(state) remains the full oldest-first pass for the dialogue harness, now defined as a loop over the same _cogitate_kline with an identity check (entry left attention → next entry slid into the index) — verified byte-identical on mhall.ks, so the refactor preserved pass semantics exactly. WorkRunner._run_work_item calls cogitate(self._state, kline) per pop; Engine submits the work-list delta to the backlog, so the backlog mirrors the work list and every entry gets exactly its own cogitation. Verified: 90 tests, mhall identical, engine smoke — 7 pops → 7 per-item cogitations, final state unchanged (residue 7, frame 8), forced emission → frame event.

*Relevance: high*
*Context: Per-item cogitation in WorkRunner*
*Tags: architecture work-runner cogitator fix*

---
*Observed: 2026-09-22T08:48:50.902Z*
