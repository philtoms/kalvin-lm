---
type: source
title: "Observation: Entry-sequence fix: no-op closer guard, row-closes-before-frame, quiet known-resolution"
tags:
  - trs
  - mhall
  - trace
  - entry-sequence
  - closers
  - no-op-guard
  - r4
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-entry-sequence-fix-no-op-closer-guard-row-closes-before-fram
relevance: high
observed_at: 2026-10-30T17:32:18.862Z
source_context: Fixing the mhall entry sequence after user review of the trace tables
---

# ⭐ Observation: Entry-sequence fix: no-op closer guard, row-closes-before-frame, quiet known-resolution

User flagged the mhall entry sequence: row 6 (E=Object) must appear between row 5's pairing and row 5's close — the depth-first descent before the BP cascade. Two defects found and fixed (commit 38a02d0): (1) PREEMPTIVE SIBLING CLOSER — the Q reentry paired both QO:Object and ALLQ:Query (both closers); ALLQ resolved instantly and resolved the Q reentry before the Object child ever popped, so the ALL row closed first and row 6 rendered late. Fix: a closer whose fit-frame is already on record is skipped — but ONLY as a pure no-op (both sides settle without children); a pairing that still carries descent work runs even when the frame exists (the first cut of this guard was too aggressive: it skipped the WDH-side Object×SVO pairing whose SV side carried the resolution, and wdmh lost grounding — a reentry whose every candidate is barred or proven must resolve quietly as already-known, else it hangs as a rows=0 dead end). (2) CLOCK ORDER — _pair minted the F==E frame before checking instant resolution, so the frame cascade's ticks (resolving waiting parents) preceded the row's own closed clock; the row now resolves first, then its frame mints. Emergent selector discovered: mirror pairings whose overfit side receives NO witness nodes (the candidate's adjacency has nothing contained in the region, e.g. Subject×MS:Subject's Mary side) push empty-adjacency children that halt R4 and never resolve — so such mirrors lose slots honestly and the goal-side pairing carries (Subject reentry slots SVO → VO×SVO, matching the user's table's second descent exactly).

*Relevance: high*
*Context: Fixing the mhall entry sequence after user review of the trace tables*
*Tags: trs mhall trace entry-sequence closers no-op-guard r4*

---
*Observed: 2026-10-30T17:32:18.862Z*
