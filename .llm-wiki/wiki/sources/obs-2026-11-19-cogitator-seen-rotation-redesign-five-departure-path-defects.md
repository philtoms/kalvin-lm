---
type: source
title: "Observation: Cogitator seen-rotation redesign: five departure-path defects fixed"
tags:
  - cogitator
  - reentry
  - rationalisation
  - bugfix
status: observation
created: 2026-11-19
updated: 2026-11-19
slug: obs-2026-11-19-cogitator-seen-rotation-redesign-five-departure-path-defects
relevance: high
observed_at: 2026-11-19T17:05:14.914Z
source_context: Fixing wdmh completing only step 1 after the push-semantics redesign
---

# ⭐ Observation: Cogitator seen-rotation redesign: five departure-path defects fixed

User redesigned cogitator push semantics (single candidate C, seen-map rotation replacing fresh()/frame-length gating). Five defects found and fixed in src/rationalisation/cogitator.py: (1) served-stall retry did stack.append(entry) on the C=None probe — re-ran S4 forever until MAX_ENTRIES capped (mhall -s spun 4098 turns); now re-pushes through the parent so candidates recompute from the grown domain. (2) _reenter rotation failure dropped the run silently — stack drained, run.finish never fired, step never recorded (wdmh "only completes step 1"); fixed by (4). (3) framed-at-root appended None — now finishes the run. (4) stall/exhaust rotated through the grandparent discarding the parent's in-progress evaluation; now every child departure (framed, stalled, exhausted) resumes the parent in place — _reenter deleted entirely; rotation is the parent's next turn reading seen. (5) elif chain skipped O/F when U's push failed; three independent ifs restored. Cycle-breaker replacing fresh(): the empty probe (C=None stall child) counts as seen (KNone sentinel) — fires once per (parent, ask); a serve re-pushes through the parent so newly added domain klines are still found. Results: mhall grounds (default and -s); wdmh completes both steps, step 2 honestly stalls on DH (did|have), does NOT ground — HEAD did ground it. docs/rationalisation.md §2.2/§2.3 REENTER/T1 updated to match.

*Relevance: high*
*Context: Fixing wdmh completing only step 1 after the push-semantics redesign*
*Tags: cogitator reentry rationalisation bugfix*

---
*Observed: 2026-11-19T17:05:14.914Z*
