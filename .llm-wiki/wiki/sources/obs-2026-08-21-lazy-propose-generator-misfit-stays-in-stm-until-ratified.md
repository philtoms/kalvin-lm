---
type: source
title: "Observation: Lazy propose generator + misfit stays in STM until ratified"
tags:
  - dialogue
  - engine
  - proposals
  - laziness
  - bfs
status: observation
created: 2026-08-21
updated: 2026-08-21
slug: obs-2026-08-21-lazy-propose-generator-misfit-stays-in-stm-until-ratified
relevance: critical
observed_at: 2026-08-21T07:43:41.849Z
source_context: Lazy generator propose redesign + misfit lifecycle fix
---

# 🔴 Observation: Lazy propose generator + misfit stays in STM until ratified

Redesigned propose into a lazy generator (BUDGET=3 per call, reentry shares it) with BFS _edge_hops frontier over every non-terminal non-identity bucket edge (min-hops first, consumers halt at k nearest) — replacing single-path selection that restricted cogitation to deterministic LTM paths. Discovery order (pivots → fills → reentry) replaces the global sort;MisfitStrategy protocol now yields Iterator[KValue]. Key lifecycle fix discovered by instrumentation: proposals never enter STM, so the S4 route's remove_stm was a silent MISS on declined proposals, and cogitate's misfit arm consumed the entry (remove_stm_at) at framing time — before the proposal's fate was known, so a declined proposal left nothing in STM to re-propose from. Now framing does not consume the misfit (removal only on ratification/grounding or unknown-ask S4), and refused filters emissions but NOT reentry sources. Result: mhall and wdmh both now propose WDMH:[Mary,had,a,little] S2 181/184 (declined), then recover with WDMH:[had,Mary,a,little,lamb] S2 244 — grading responds to W:[Object] grounding via chain-meet crossovers in _grade (_chain + entity-side sub-canon chains). Also learned: DH:[did,have] is NOT a canon (sig([did,have]) != DH) so grouped sub-canon resolution never fired in mhall — the 5-node form always came from reentry completing the greedy pivot proposal with 'lamb'. Harness trace: proposals carry band + decimal significance byte (e.g. 'S3 52'); identity asks display bare.

*Relevance: critical*
*Context: Lazy generator propose redesign + misfit lifecycle fix*
*Tags: dialogue engine proposals laziness bfs*

---
*Observed: 2026-08-21T07:43:41.849Z*
