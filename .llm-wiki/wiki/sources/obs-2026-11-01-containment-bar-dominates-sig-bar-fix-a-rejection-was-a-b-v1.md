---
type: source
title: "Observation: Containment bar dominates sig-bar — fix-A rejection was a B-v1 confound"
tags:
  - rationalisation
  - completeness
  - in-path
  - containment-bar
  - churn
status: observation
created: 2026-11-01
updated: 2026-11-01
slug: obs-2026-11-01-containment-bar-dominates-sig-bar-fix-a-rejection-was-a-b-v1
relevance: critical
observed_at: 2026-11-01T09:12:55.696Z
source_context: Pursuing the measured completeness gap — why relaxing the path bar broke mhall
---

# 🔴 Observation: Containment bar dominates sig-bar — fix-A rejection was a B-v1 confound

Pursued the completeness gap — the earlier fix-A rejection was CONFOUNDED. Fix A (kline-comparison in _in_path) was originally tested under B v1 (any turn-away halts); re-tested on top of B v2, mhall grounds fine. The real cost of fix A: wdmh scenario 0 (MHALL→SVO with wdmh priming) loses grounding STRUCTURALLY — the admitted same-sig pairings (held MHALL claim vs the MHALL-region reentry) spawn re-derivation churn: residual chains oscillating to the depth bound, 535/602 rows left open, closing cascades never fire. Not budget: fails at every MAX_DEPTH up to 30; closers-first doesn't rescue it. The sig-bar was load-bearing against exactly this churn. THE REFINED RULE (working tree, not landed): _in_path bars a candidate when its SIDE is CONTAINED in an ancestor entry's side (side(c) & side(a) == side(c)) — churn (held kline's content ⊆ ancestor's content) is suppressed; genuinely new content still pairs. MEASURED (same 400-scenario sweep, refusals identical at 77 across all regimes — bar-independent as expected): sig-equality 245 grounded / 78 missed; kline-equality 270/53 but wdmh-s0 broken; CONTAINMENT + insertion order 255/68 all-green; CONTAINMENT + closers-first sort (the dormant commented-out sorted() in _reenter) 261/62 all-green with fewer rows (mhall 172, wdmh 109 vs 602 under fix A). Containment strictly dominates the sig-bar: +10-16 groundings, corpus intact, churn gone. Remaining 62 misses attributable to the checker's other liberties (global frames, no chain restriction, no bounds, exchange loosening). Both variants keep the suite green (176 passed, 17 xfailed) with certificates verified. Also noted: user's 8e582f5 renamed TrsRunner→Runner; my b525f7d commit swept their staged work.

*Relevance: critical*
*Context: Pursuing the measured completeness gap — why relaxing the path bar broke mhall*
*Tags: rationalisation completeness in-path containment-bar churn*

---
*Observed: 2026-11-01T09:12:55.696Z*
