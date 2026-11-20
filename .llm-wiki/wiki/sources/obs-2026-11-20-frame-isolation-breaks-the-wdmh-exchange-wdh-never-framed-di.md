---
type: source
title: "Observation: Frame isolation breaks the wdmh exchange — WDH never framed, diagnosis complete"
tags:
  - cogitator
  - framing
  - exchange
  - frame-isolation
  - wdmh
status: observation
created: 2026-11-20
updated: 2026-11-20
slug: obs-2026-11-20-frame-isolation-breaks-the-wdmh-exchange-wdh-never-framed-di
relevance: high
observed_at: 2026-11-20T09:39:18.257Z
source_context: Frame isolation exploration — wdmh exchange never frames WDH
---

# ⭐ Observation: Frame isolation breaks the wdmh exchange — WDH never framed, diagnosis complete

Diagnosed why frame isolation (frames moved from _Run to Entry, staged diff on cogitator.py) grounds mhall but not wdmh. Probe dev/probes/probe_frame_isolation.py logs instance-level frames. Three linked causes: (1) The exchange frames the signature only in the CURRENT stalling entry — `_pop(witness)` reframes a copy of the witness's parent with the WITNESS's frame, and a stalled probe framed nothing, so the witness's parent resumes empty (violates §2.3 "the witness's parent resumes, its component framed"; was automatic under the run-level frame). (2) A component is minted as a frame element only by that ask's own S2; WDH's turns all precede the exchange (U-side explores and drains first), and afterwards the root cannot reenter WDH — seen exhausted (only candidate what:Object), no other overlapping signature — so no WDH S2 ever fires. (3) At the post-exchange root turn the frame {Object, what, had, did|have, ALL, HALL, Mary} COVERS WDH exactly (what ∪ did|have), but `framed(x) = any(contained(x, f))` demands one element ⊇ WDH, so u_ok fails. Experimentally (dev/probes/probe_frame_isolation_fix.py, monkeypatched): witness.frame(A) alone, merge-reframe alone, and both together all still ungrounded; union-coverage framed() (x ⊆ ⋃frame) alone grounds wdmh (claim WDMH:[MHALL] step 1, mhall unchanged) — matching §2.2's "frame, covered content" reading. Also found: the stale witness-parent copy's departure REPLACES the live root's frame with {} (violates §2.4 "frames only grow"), and the staged root-tail run.finish fires on every unframed root-copy departure → steps=3 → test_a_served_stall_supplies_scaffolding_on_demand red (assert 2), also in tension with §2.3 WALK "a root that never aligns never finishes".

*Relevance: high*
*Context: Frame isolation exploration — wdmh exchange never frames WDH*
*Tags: cogitator framing exchange frame-isolation wdmh*

---
*Observed: 2026-11-20T09:39:18.257Z*
