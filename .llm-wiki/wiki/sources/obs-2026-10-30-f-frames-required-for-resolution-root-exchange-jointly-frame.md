---
type: source
title: "Observation: F frames required for resolution; root exchange jointly frames the parents' fits"
tags:
  - trs
  - framing
  - root-exchange
  - fits
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-f-frames-required-for-resolution-root-exchange-jointly-frame
relevance: high
observed_at: 2026-10-30T11:27:39.537Z
source_context: Resolving how F components frame — the root exchange frames the parents' fits
---

# ⭐ Observation: F frames required for resolution; root exchange jointly frames the parents' fits

Answered the had-framing question with instrumentation and closed the semantic gap. Findings: (1) had was NEVER framed in the old run — resolution only required U∧O, not F, violating the rule \"resolved when F, U and O are all framed\"; (2) even opportunistically, (had × hadVerb)'s F==E frame was suppressed by _frame's global covered-check (had ⊆ HALL frame) although chain-excluded rows (the had:DH row, same reentry as the HALL frame's source) needed it — _frame now always records a witnessed value (dedup only by identity). Three changes: _satisfied requires _covered(row.F, chain); the root exchange frames the two parent FITS (a.parent.F and b.parent.F — \"WDH for WH\": what for had) as JOINT frames (frame_re=None — a mutual witness belongs to neither subtree; attributing to one parent made the chain exclusion block that very parent); _frame gained joint=. Verified: mhall+wdmh claims ground; counterfactual removing ONLY the joint fit frames → wdmh claim False (what frames nowhere else — the root exchange is load-bearing). 134 tests pass.

*Relevance: high*
*Context: Resolving how F components frame — the root exchange frames the parents' fits*
*Tags: trs framing root-exchange fits*

---
*Observed: 2026-10-30T11:27:39.537Z*
