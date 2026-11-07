---
type: source
title: "Observation: End vocabulary: rationalised (ask, outcome) vs exhausted (construction, halt)"
tags:
  - cogitator
  - ends
  - vocabulary
  - proposals
status: observation
created: 2026-11-07
updated: 2026-11-07
slug: obs-2026-11-07-end-vocabulary-rationalised-ask-outcome-vs-exhausted-constru
relevance: medium
observed_at: 2026-11-07T15:59:33.282Z
source_context: Distinguishing ask completion from construction exhaustion in end vocabulary
---

# 🔍 Observation: End vocabulary: rationalised (ask, outcome) vs exhausted (construction, halt)

End vocabulary now distinguishes the run kinds (user ruling: distinguish rationalisation — the ask — from the construction kind before updating the vocab). In cogitator.py: self.goals (per run) collects each depth-1 close with candidate sig == goal.sig — this run's witnesses of the claim; at the root drain, an ask run (goal is not None) with goals achieved ends with End("rationalised: N goal path(s)", None) — an outcome, kline None so proposals() files nothing; zero goals or goal-less construction runs keep "exhausted: no candidate resolved both sides" with the entry kline → honest acquire proposal. harness.py _render_step renders kline-less ends as "end {reason}" instead of "halt {reason} — kline". Verified: mhall ask → "end rationalised: 1 goal path"; wdmh ask → "end rationalised: 3 goal paths"; construction-kind drain (candidate that fails to ground) → "halt exhausted..." + acquire proposal; degenerate no-candidate construction root parks at loop top → connect proposal. Tests 162/15/16; no bogus proposals from completions (wdmh pending stays connects only).

*Relevance: medium*
*Context: Distinguishing ask completion from construction exhaustion in end vocabulary*
*Tags: cogitator ends vocabulary proposals*

---
*Observed: 2026-11-07T15:59:33.282Z*
