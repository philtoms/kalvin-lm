---
type: source
title: "Observation: I-1 enforcement code removed; no-neighbours halt is the ask"
tags:
  - trs
  - i-1
  - enforcement
  - no-neighbours
  - presentation
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-i-1-enforcement-code-removed-no-neighbours-halt-is-the-ask
relevance: medium
observed_at: 2026-10-06T12:50:00.295Z
source_context: Removing redundant I-1 enforcement code from the runner
---

# 🔍 Observation: I-1 enforcement code removed; no-neighbours halt is the ask

REMOVAL of redundant I-1 code (user ruling): 'held is no longer an invariant so no need to uphold it in code.' Deleted the per-pop unrecognised-node check from rationalise() (the held-set + per-node 'R4: unrecognised node' ends) — the last I-1 enforcement remnant. The no-neighbours route carries BOTH consistency and the ask: from-empty entries now halt once with 'no candidates — the entry terminates', and proposals() maps that end to a connect proposal for the entry itself (verified: wdmh step 1 = single halt, single connect ask; no per-node ask flood). Spec: Definition 8 now 'Presentation admits any kline — the system never seeds and checks nothing at the boundary'; Definition 3's trailing 'asked for at the boundary' sentence removed (no longer true at presentation; asks arise from no-neighbours halts via Definition 15). Harness docstring reworded. connect-from-empty test back to single connect proposal. 156 passing. History: amendment 19's per-node S4 ask generation is now fully withdrawn — the sequence of rulings (19: route per-node; 20: I-1 out; now: enforcement code out) ended at the simplest model: presentation checks nothing, R1 is sole admission, nothing seeds. Uncommitted.

*Relevance: medium*
*Context: Removing redundant I-1 enforcement code from the runner*
*Tags: trs i-1 enforcement no-neighbours presentation*

---
*Observed: 2026-10-06T12:50:00.295Z*
