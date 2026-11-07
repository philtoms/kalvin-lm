---
type: source
title: S4 "exhaustion" was a backtrack bookkeeping defect
status: insight
category: debugging
created: 2026-11-07
updated: 2026-11-07
slug: s4-backtrack-exhaustion-diagnosis
---

# S4 "exhaustion" was a backtrack bookkeeping defect

Fixed in [[src/rationalisation/cogitator.py]] (see obs-2026-11-07 S4-backtrack-fix). Symptom: wdmh.ks step 4 halts "exhausted: no candidate resolved both sides" with no claim. Diagnosis path that worked: (1) copy the module to an instrumented twin, add PUSH/POP/PAIR/CLOSE/MINT trace lines with a LABELS map built from the harness's word_bits, drive the exact failing scenario by replicating Runner.run's prime-then-step sequence and swapping in the debug cogitator at the failing step; (2) the trace names the mechanism — POP branch-B printed "DISCARD caller candidate" and later candidates printed "SKIP-O(seen)". Lessons: an exhaustion halt can be a *bookkeeping* artifact, not a domain gap — the pairing that grounds the claim was never re-evaluated because side-descent memory (seen) had no invalidation condition. The invalidation that fits this architecture: frame growth (len(self.frame)) — a descended side re-descends only when new frames minted since its last descent; that is both sufficient to ground WDMH and a progress condition for termination. Second lesson: when two return paths from a child exist (stall-return vs drain-return), they should perform the SAME parent continuation (next), with per-candidate descent memory reset only at candidate completion — anything else leaks state across candidates.

*Category: debugging*

---
*Captured: 2026-11-07*

## Related

_Add links to related pages._
