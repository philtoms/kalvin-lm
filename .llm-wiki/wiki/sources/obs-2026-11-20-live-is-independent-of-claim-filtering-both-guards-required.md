---
type: source
title: "Observation: live is independent of claim filtering — both guards required"
tags:
  - cogitator
  - live
  - finish
  - semantics
status: observation
created: 2026-11-20
updated: 2026-11-20
slug: obs-2026-11-20-live-is-independent-of-claim-filtering-both-guards-required
relevance: medium
observed_at: 2026-11-20T07:40:01.568Z
source_context: Questioning whether the live counter is still needed after claim filtering
---

# 🔍 Observation: live is independent of claim filtering — both guards required

User asked whether first-past-the-post claim filtering makes _Run.live removable. Demonstrated no (temporarily ungated both finish sites): wdmh renders 3 step headers — the double is NOT a second broadcast (the claims set held; one claim in the sink) but end_step firing twice, two Step tables each rendering the same single claim. The two guards are independent: `claims` guards the broadcast channel against duplicates; `live` guards the run's END — the witness-notify puts a second live entry on the stack, so the root can tail-depart early while the residue (WDH×what) is still queued, finishing the run before the witness chain completes: the report leaves early, the claim grounds after the run's end, and the second root departure finishes again. Alternatives to live are worse: a done-flag finish leaves residue running post-finish (aliased events list mutates the delivered report; claim grounds post-run); a zombie gate on finished runs loses the claim entirely; one-push-per-turn at the exchange restores the old invariant but is lossy (dropped chain's burned candidates never re-pair). live is minimal: one int, 5 increment sites (every run-entry append), 1 decrement (turn start), 2 read sites (root departures).

*Relevance: medium*
*Context: Questioning whether the live counter is still needed after claim filtering*
*Tags: cogitator live finish semantics*

---
*Observed: 2026-11-20T07:40:01.568Z*
