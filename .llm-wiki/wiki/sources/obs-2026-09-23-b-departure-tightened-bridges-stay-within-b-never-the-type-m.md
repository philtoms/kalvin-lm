---
type: source
title: "Observation: B-departure tightened: bridges stay within B, never the type — mhall proposes the walked answer"
tags:
  - def15
  - b-departure
  - type-not-slot
  - shortcut
  - stm-bridges
  - walk-coverage
  - mhall
  - wdmh
status: observation
created: 2026-09-23
updated: 2026-09-23
slug: obs-2026-09-23-b-departure-tightened-bridges-stay-within-b-never-the-type-m
relevance: critical
observed_at: 2026-09-23T14:47:44.083Z
source_context: Fixing the invalid MarySubject:[SVO] bridge
---

# 🔴 Observation: B-departure tightened: bridges stay within B, never the type — mhall proposes the walked answer

Root cause of the MarySubject:[SVO] invalid bridge found and fixed — Phil's instinct ("shortcuts are always a clue") confirmed twice over. The bridge was NOT composed under goal=SVO: instrumented _meet/_ground_composed/extend_stm and caught the provenance chain — Subject:[SVO] was composed under goal=lamb (A=MarySubject:[Subject], B=lamb:[Object]) because _walk's B-departure admission used `residual(excess, v)==0` (v ⊇ excess, any superset) — the SVO canon head's merged content {S,V,O} ⊇ {Object} qualified, so B departed from a value carrying content OUTSIDE the relationship entirely; the bridge rode STM (extend_stm) into later derivations, which forward-replaced [Subject]→[SVO] (one-move done, σ-equal, claim-blind). The doc's own wording ("B's walk departs the held value containing the overfit") licensed supersets — too loose. TIGHTENED (doc Def 15 + derivation.py): (1) B departs a held value OF ITS OWN — residual(excess,v)==0 AND residual(v,σ(ν_B))==0 (a node or compound of ν_B containing the overfit, never content beyond B); (2) a bridge's B-side slot is never B's own head — the type is not a slot, a witness arriving at the type collapses the whole question into one node. ALSO fixed walk_closure from my first implementation: Def 15's A-side step restrictions (single-node descents) had silenced wdmh — selection coverage is UNDIRECTED bipartite reachability (node ~ kline iff node ∈ witness OR σ(head) ∋ node), Def 15 restrictions bind derivation steps only. RESULTS: mhall proposes MHALL:[Subject, Verb, Object] S1 255 via MHALL:[Mary,had,ALL]→SVO done j1=1.000 (the walked structure, pure selection, no shortcut); wdmh intact WDMH:[ALL,had,Mary] S1 255; the scaffold cascade shrank to ONE legitimate self-derivation (ALLQuery:[Object] via [Query]→[Object], declined at S4, no follow-on composition); pure-overfit scaffold derivations now honestly stuck (the appendix ruling holds); 90/90 tests; test_selection_orders_goals_by_overlap updated to the new law (a,l covered by witness path w→[o]→all:[o]→all→[a,l,l] — [DH, MHALL] order preserved). Probes: dev/dialogue/probe_svo_shortcut.py (instrumented run/meet/extend_stm).

*Relevance: critical*
*Context: Fixing the invalid MarySubject:[SVO] bridge*
*Tags: def15 b-departure type-not-slot shortcut stm-bridges walk-coverage mhall wdmh*

---
*Observed: 2026-09-23T14:47:44.083Z*
