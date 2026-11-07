---
type: source
title: "Observation: Trace elision removed — every reentry logged; grounding-path pruning deferred"
tags:
  - trs
  - trace-table
  - harness
  - elision
  - ruling
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-trace-elision-removed-every-reentry-logged-grounding-path-pr
relevance: high
observed_at: 2026-11-03T13:33:32.627Z
source_context: "Removing trace elision per user ruling: every reentry logged, prune later"
---

# ⭐ Observation: Trace elision removed — every reentry logged; grounding-path pruning deferred

User ruling on trace rendering: elision removed from the harness event table (dev/dialogue/harness.py _render_events) — minting is not a visibility filter. Every reentry is now logged through its full pairing pass (stalls/exchanges/closes as before), so wdmh step 4 opens with HLAL's complete 8-pairing pass followed by the DH stall. Table grows: mhall 148→281 rows, wdmh 420→676. Planned NEXT (user): prune entries that do not lead directly to grounding, at a later step — a counterpart of dev/dialogue/trace.py's optimal-path trace for the ReferenceCogitator ("one row per reentry — paths stay orthogonal"). Tests unchanged: 162/15/16. Pair+close still render as two rows for a grounding pairing (event structure, not elision) — pruning may merge them.

*Relevance: high*
*Context: Removing trace elision per user ruling: every reentry logged, prune later*
*Tags: trs trace-table harness elision ruling*

---
*Observed: 2026-11-03T13:33:32.627Z*
