---
type: source
title: "Observation: Exchange closure is the frame alone; child-resolution channel excluded for stalls"
tags:
  - trs
  - exchange
  - redundancy
  - frame-channel
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-exchange-closure-is-the-frame-alone-child-resolution-channel
relevance: high
observed_at: 2026-10-30T11:05:25.344Z
source_context: Exchange channel redundancy found by user commenting out _admit/_frame
---

# ⭐ Observation: Exchange closure is the frame alone; child-resolution channel excluded for stalls

User's counterfactual exposed a channel redundancy in the exchange: commenting _admit and _frame (524/532) still grounded wdmh — closure flowed through _side_ok's child-resolution channel (a.resolved=True + _after(parent)) which duplicated the frame channel. Per the frame rules (\"U and O are framed iff U==O and U==C — the stalled S4s\") the frame must be the load-bearing channel. Fix: _side_ok's child-resolution channel now excludes stalled children (ch.stalled is None — a stalled reentry has no rows, nothing resolved it); _exchange dropped its _after(parent) calls — the _frame cascade propagates. Verified: both claims ground; counterfactual neutralizing _frame inside _exchange only → wdmh claim False (the frame is now provably load-bearing). _admit (the R1 store DH:[did,have]) remains non-load-bearing for current-run closure — correct: R1 stores are learning for future runs. Lesson: when two mechanisms can both close a side, comment-out probes are the way to find which one is real; keep the rule-mandated one.

*Relevance: high*
*Context: Exchange channel redundancy found by user commenting out _admit/_frame*
*Tags: trs exchange redundancy frame-channel*

---
*Observed: 2026-10-30T11:05:25.344Z*
