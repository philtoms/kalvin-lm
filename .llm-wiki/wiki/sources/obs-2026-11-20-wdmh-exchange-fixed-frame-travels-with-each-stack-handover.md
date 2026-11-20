---
type: source
title: "Observation: wdmh exchange fixed — frame travels with each stack handover"
tags:
  - cogitator
  - framing
  - exchange
  - frame-isolation
  - wdmh
status: observation
created: 2026-11-20
updated: 2026-11-20
slug: obs-2026-11-20-wdmh-exchange-fixed-frame-travels-with-each-stack-handover
relevance: high
observed_at: 2026-11-20T09:56:16.814Z
source_context: Verifying and cleaning up the frame-isolation exchange fix
---

# ⭐ Observation: wdmh exchange fixed — frame travels with each stack handover

The wdmh frame-isolation break is resolved. The fix (Phil's, staged on cogitator.py): `_pop(parent, entry)` takes the destination explicitly and the exchange calls `self._pop(witness.parent, entry)` — the witness's parent resumes reframed with the CURRENT entry's frame, not the witness's stale one. Mechanism: the frame is a token that travels with each stack handover (push: parent→child; departure: departing entry→whoever resumes; exchange: current entry→witness's parent); no turn ever reads a suspended instance's frame, so the stale-snapshot clobber cannot occur. Why the exchange handover is sound: the common-ancestor precondition guarantees the current entry's frame already holds the witness side's knowledge (the U-side drains into the ancestor before the O-side is pushed and inherits it), so the witness's parent receives "everything known at the meet point + the exchanged signature" in one forward pass — its next turn has all three components framed, S2s, and mints WDH itself (probe shows frame => {Object,what,had,did|have,what|did|have}), which propagates to the root's claim WDMH:[MHALL]. Also removed the staged root-tail `run.finish(run)` (fired per unframed root-copy departure → steps=3, duplicate Step tables; reported ungrounded runs, breaking test_an_unserved_stall_stands_silent and contradicting §2.3 WALK) — 163 passed. docs/rationalisation.md §2.2/§2.3 updated: frame moved from run state to entry state, forward-pass handover described, exchange described as handing the current entry's frame. Side-finding: union-coverage framed() variants (from the earlier experiment) now produce a duplicate claim + second exchange from post-claim leftovers — containment + forward handover is the cleaner semantics. Probes: dev/probes/probe_frame_isolation.py (instance-level frame tracer), probe_frame_isolation_fix.py (variant matrix, updated to new _pop signature).

*Relevance: high*
*Context: Verifying and cleaning up the frame-isolation exchange fix*
*Tags: cogitator framing exchange frame-isolation wdmh*

---
*Observed: 2026-11-20T09:56:16.814Z*
