---
type: source
title: "Observation: Selection trace live: supplied goals stall at entry, selected goals canonicalise, supplied finishes"
tags:
  - traceability
  - candidates
  - supplied-selected
  - mhall
  - wdmh
  - harness
status: observation
created: 2026-09-23
updated: 2026-09-23
slug: obs-2026-09-23-selection-trace-live-supplied-goals-stall-at-entry-selected-
relevance: critical
observed_at: 2026-09-23T12:55:09.658Z
source_context: Making engine selection traceable in the harness presentation
---

# 🔴 Observation: Selection trace live: supplied goals stall at entry, selected goals canonicalise, supplied finishes

Selection traceability implemented: scored_candidates (hop.py) returns (KLine, source) pairs — "supplied" (the => construct's goal, ahead of selection) vs "selected" (Def 22 coverage, γ-ordered); candidate_goals becomes a wrapper (probes untouched). DerivationResult gains goal/goal_source; Hop.run stamps them; cogitate(state, collect=...) threads (queued, results) pairs out; harness Turn.derivations + renderer prints "derives <entry> → <goal> (source) ending j1=..." lines with per-turn dedupe (cogitate re-passes duplicate). 90/90 tests. THE FINDING the trace makes inspectable — identical shape in both scripts: (1) the supplied goal STALLS AT ENTRY (mhall: MHALL→SVO supplied stuck j1=0.000; wdmh: WDMH→MHALL-canon supplied stuck j1=0.400); (2) a legitimately SELECTED goal (Def 22 coverage) performs the canonicalisation and changes the re-entry state (mhall: MHALL→ALL-canon selected stuck j1=0.600 — canonicalised [a,little,lamb]→ALL; wdmh: WDMH→had:[did,have] selected — did→had, re-entry [what, had, Mary]); (3) the supplied goal then reaches done on the prepared state (both done j1=1.000 → proposals). So: the arrangement work rides the supplied goal; Def 22 selection alone would not produce it (SVO disjoint from the ask's nodes — verified ∅×5); the selected goals contribute content-neutral canonicalisation only. The scaffolds' derivations also visible: MarySubject→SVO stuck j1=0.333 (the pure-overfit wall, appendix case), QueryObject↔lamb:[Object] done-without-moving (ground-path dones, correctly filtered from proposals). This gives Phil the instrument to judge whether the supplied-goal mechanism is the right reason or a stand-in — the §13 "supplies a goal" / Def 21 wording fork.

*Relevance: critical*
*Context: Making engine selection traceable in the harness presentation*
*Tags: traceability candidates supplied-selected mhall wdmh harness*

---
*Observed: 2026-09-23T12:55:09.658Z*
