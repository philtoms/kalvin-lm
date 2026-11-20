---
type: source
title: "Observation: Real-stack Cogitator verified behaviour-identical"
tags:
  - cogitator
  - rationalisation
  - stack
status: observation
created: 2026-11-18
updated: 2026-11-18
slug: obs-2026-11-18-real-stack-cogitator-verified-behaviour-identical
relevance: critical
observed_at: 2026-11-18T15:22:38.581Z
source_context: Overlaying the pseudo stack onto a real stack in the Cogitator
---

# 🔴 Observation: Real-stack Cogitator verified behaviour-identical

src/rationalisation/cogitator.py rewritten onto a real stack, replacing the user's staged sketch. Discipline: the stack's TOP entry runs (peek, not pop-to-register); _push appends the child and leaves the parent suspended beneath (its candidate kept for re-evaluation); pop backtracks. All run state moved into _State (ask, goal, entries counter, frame, stalled, events) instantiated fresh per open(); stalled/events are no longer reset fields on the Cogitator — the Cogitator is now a stateless turn-taker. chain stays a per-entry immutable snapshot (frozenset computed at push from parent's chain). New API: open(ask, goal) -> root entry, turn(stack) -> bool (False = cap halt), finish(st) -> Rationalisation; rationalise = one-shot drive. Verification: 159 passed + 16 xfailed; mhall and wdmh harness traces byte-identical to 661113af HEAD (both claims intact: MHALL:[SVO], WDMH:[MHALL]). Interleaving demonstrated by dev/probes/interleave_stack.py. docs/rationalisation.md §2.2/§2.3 REENTER/T2 updated. NOT YET COMMITTED — working tree holds the rewrite; the git index still holds the user's earlier sketch.

*Relevance: critical*
*Context: Overlaying the pseudo stack onto a real stack in the Cogitator*
*Tags: cogitator rationalisation stack*

---
*Observed: 2026-11-18T15:22:38.581Z*
