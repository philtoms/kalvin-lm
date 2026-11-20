---
type: source
title: "Observation: Shared-stack cogitate verified; sink-reentrancy pop bugs fixed"
tags:
  - cogitator
  - harness
  - interleaving
  - threading
status: observation
created: 2026-11-18
updated: 2026-11-18
slug: obs-2026-11-18-shared-stack-cogitate-verified-sink-reentrancy-pop-bugs-fixe
relevance: critical
observed_at: 2026-11-18T16:19:12.855Z
source_context: Verifying the harness-pushes-steps-onto-cogitator-stack arrangement
---

# 🔴 Observation: Shared-stack cogitate verified; sink-reentrancy pop bugs fixed

The harness→cogitator relationship now supports true interleaving: rationalise(ask, goal, end_step) pushes a run's root onto the cogitator's SHARED stack (the stack lives on the Cogitator); cogitate() drains the whole stack. _State gained a finish callback; run completion = the root departing (entry.parent is None) or the cap — NOT stack-empty. Three bugs found and fixed in the user's version: (1) finish fired only at stack-empty — with coexisting runs every run but the last to drain lost its end_step (run2's report vanished); (2) cap-halt did finish+break leaving the capped entry on the stack — the next cogitate re-finished it (double end_step, duplicate Step) and the break abandoned runs below; (3) FATAL for the threaded intent: stack.pop() in the stall/bottom paths assumed the current entry is the top, but a sink (on_claim) can land another run's root mid-iteration — the pop discarded the WRONG entry (the preemptor's root, silently lost), the finished root zombie-re-ran as a stall, appending a spurious event into the already-delivered live Rationalisation(st.events) list. Fixes: _drop(entry)/_drop_run(st) remove by identity wherever the entries sit (in-place del, no list rebind — a rebind would lose concurrent appends from a harness thread); cap continues the loop instead of break. Tests ported to push+cogitate (3 call sites); end_step optional like other sinks. Verified: 159 passed + 16 xfailed; mhall/wdmh byte-identical to HEAD; probe dev/probes/interleave_stack.py — LIFO both orders, mid-run preemption via claim sink, cap with double cogitate: every run's stream identical to solo, exactly one finish each, zero residue. Thread contract for the future: ONE cogitate thread at a time; sinks run on the cogitate thread; cogitate exits on empty stack so a threaded harness must not race a push past the loop exit.

*Relevance: critical*
*Context: Verifying the harness-pushes-steps-onto-cogitator-stack arrangement*
*Tags: cogitator harness interleaving threading*

---
*Observed: 2026-11-18T16:19:12.855Z*
