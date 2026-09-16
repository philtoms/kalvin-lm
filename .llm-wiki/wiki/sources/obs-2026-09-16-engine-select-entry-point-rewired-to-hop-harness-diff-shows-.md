---
type: source
title: "Observation: Engine select entry point rewired to Hop; harness diff shows junk proposals eliminated"
tags:
  - kalvin
  - engine
  - integration
  - hop
  - selection
  - harness
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-engine-select-entry-point-rewired-to-hop-harness-diff-shows-
relevance: critical
observed_at: 2026-09-16T12:50:53.641Z
source_context: Wiring the hop layer into dialogue/engine.py as the select entry point
---

# 🔴 Observation: Engine select entry point rewired to Hop; harness diff shows junk proposals eliminated

Integrated the hop layer into the dialogue engine as the new select entry point, per Phil's expectation. src/dialogue/engine.py: `self._misfit = Derivation(state)` (dialogue.derivation port) and `_select` (S2-band goal candidates, misfit-mass order) DELETED; replaced by `_propose(kline)` = `Hop(self._held(), kline, signifier).run()` — Def 22 goal list in order, per-goal Def 23 trawl, multiple derivations per hop; done results with trace ≥2 become proposals (KValue, gamma byte, refusal check) exactly as before. `_held()` = state.where(non-terminal) + self._writes (engine-held write backlog — the progressive path: hop writes feed later hops' trawls; promoting writes into EngineState is future re-entry work). VERIFIED via dialogue harness on all three .ks scripts, old-vs-new diff: wdmh-underfit-o byte-identical; mhall and wdmh-underfit eliminate the old engine's junk S4-declined proposals (MarySubject:[S,V,O,SVO], WDMH:[WDMH], MHALL:[MHALL] etc.) and their junk groundings, while every legitimate grounding (lamb, ALL:[a,little,lamb], MHALL, SVO) survives identically — asks S2=4→0. The junk came from the old S2-gate proposing degenerate done-proposals the supervisor auto-declined; the hop's coverage-pool + J-order + frozen scope never fires them. 51 tests pass. dialogue.derivation module remains (unused by engine) — candidate for removal once reentry.py is rewired. Uncommitted.

*Relevance: critical*
*Context: Wiring the hop layer into dialogue/engine.py as the select entry point*
*Tags: kalvin engine integration hop selection harness*

---
*Observed: 2026-09-16T12:50:53.641Z*
