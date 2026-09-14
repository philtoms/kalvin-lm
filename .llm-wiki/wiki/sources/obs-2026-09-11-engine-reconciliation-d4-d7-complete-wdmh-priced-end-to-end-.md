---
type: source
title: "Observation: Engine reconciliation D4–D7 complete; WDMH priced end-to-end in engine"
tags:
  - engine
  - reconciliation
  - d4
  - d5
  - d6
  - d7
  - pivot-fill
  - selection
  - acq-depth
status: observation
created: 2026-09-11
updated: 2026-09-11
slug: obs-2026-09-11-engine-reconciliation-d4-d7-complete-wdmh-priced-end-to-end-
relevance: critical
observed_at: 2026-09-11T21:28:35.366Z
source_context: Completing D4-D7 engine reconciliation with commits between tasks
---

# 🔴 Observation: Engine reconciliation D4–D7 complete; WDMH priced end-to-end in engine

Completed the engine↔ks2 fourth-pass reconciliation (D4-D7; D1 ASK_BPE_TOKEN skipped per user, D8 closed — experimental code, no derivation executor, pivot_fill stays). Commits on algebra: 3932f4f (D4: KLine.acq_depth — the acquisition record; identity ignores it; expand prices ungrounded matched slots by the carried record, grounding flattens; EngineState snapshots round-trip it as an optional third element), 57fce43 (D5: is_connotation split — module fn renamed is_relationship (the 1:1 shape, CONTEXT.md's Relationship); canonical band-true species added, overlap-based per Def 8: is_connotation=case 4 uncovered S3, is_denotation=case 6 covered gap-only S2, covered-with-excess shapes are neither; harness._structure_class uses the canonical split; EngineState.is_connotation delegates (containment→overlap) and is_countersignable is band-true), fc79c02 (D6: selection by occurrence — expand_fit/pivot_fill _candidates now require the candidate's signature to occur in one of the entry's nodes (bit-space containment), not word-overlap; identity inert; empty entry selects nothing; the WDMH property verified: w:[o] and dh:[d,h] selectable from wdmh:[w,dh,m], mhall never), eb067c6 (D7: significance.misfit_mass = |σ(ν_A) Δ σ(ν_B)|; pivot arm's consume licence — proposals must strictly shrink the misfit mass vs their pivot; crossover-fill arm same licence vs the entry signature; gap fills priced by their REAL correspondence chain (lone-gap grouped fill priced by path to the leftover group's sig — WDMH: W→O→Q→ALL = 3; adjacency fill with no path = _MAX_HOP); proposal klines carry acq_depth = max per-node cost, filling D4's write sites; _align_to_grounded forwards the record). Test suite now 35 passing (test_gamma 25 + test_relationship 4 + test_selection 4 + test_pivot 2). Notable: D7's WDMH test reproduces the ks2 §9 pricing exactly — shared M costs 0, DH→H costs 1 (denotation edge), grouped fill costs 3 (chain) — matching the doc's Ĥ=3 example. Remaining known divergences: cogitator._candidates still overlap-based (proposal arm, left for when that arm is reconciled); reentry.py (the older S2 strategy) untouched; kalvin-symbolic.md still stale.

*Relevance: critical*
*Context: Completing D4-D7 engine reconciliation with commits between tasks*
*Tags: engine reconciliation d4 d5 d6 d7 pivot-fill selection acq-depth*

---
*Observed: 2026-09-11T21:28:35.366Z*
