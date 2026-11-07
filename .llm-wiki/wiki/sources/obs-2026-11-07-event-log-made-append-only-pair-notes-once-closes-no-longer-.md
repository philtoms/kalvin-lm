---
type: source
title: "Observation: Event log made append-only: pair notes once, closes no longer lost"
tags:
  - cogitator
  - events
  - trace
  - append-only
status: observation
created: 2026-11-07
updated: 2026-11-07
slug: obs-2026-11-07-event-log-made-append-only-pair-notes-once-closes-no-longer-
relevance: high
observed_at: 2026-11-07T13:10:38.911Z
source_context: Fixing duplicate pair rows and lost close rows in the harness trace
---

# ⭐ Observation: Event log made append-only: pair notes once, closes no longer lost

Fixed the Cogitator's event bookkeeping (user's diagnosis: "the simplistic events.copy() pattern at next candidate is the smoking gun"). Two symptoms from one defect: the loop kept TWO lists in play — self.events (pair/close appends) and call.events (stall/exchange appends) — aliasing/unaliasing across next()'s caller.events=c.events and the bottom self.events=call.events.copy(). (1) DUPLICATE rows: on a stall-return, self.events was still the same list object as the parent's, so the post-backtrack re-pair appended a second identical pair event (mhall's doubled `ALL × a:Det` row). (2) LOST rows: on a drain-return, the child's bottom copy made self.events a new list while next() handed the caller the pre-copy list — the caller's post-descent CLOSE appended into the orphaned copy and vanished (mhall's missing `Verb × SVO resolved`; the entire drain cascade Verb→SV→Object→Query→ALL closes was invisible). Both violate docs/rationalisation.md §2.4 "the events list only appends". Fix in src/rationalisation/cogitator.py: ONE append-only self.events per run — _Call.events field deleted, next() no longer reassigns events, push() takes no events param, the bottom copy line deleted, stall/exchange append to self.events; and the pair note emits only on a candidate's FIRST evaluation (guard `elif not call.at` — re-evaluation after a descent is the re-surface mechanism, not a new pairing). A pairing now traces at most twice: pair (open) then close (fate) — e.g. wdmh's `MH × had:did have` pair → `DH exchange` → `MH × had:did have … MH resolved`. Both scripts ground identically (MHALL:[SVO], WDMH:[MHALL]); tests 162/15/16.

*Relevance: high*
*Context: Fixing duplicate pair rows and lost close rows in the harness trace*
*Tags: cogitator events trace append-only*

---
*Observed: 2026-11-07T13:10:38.911Z*
