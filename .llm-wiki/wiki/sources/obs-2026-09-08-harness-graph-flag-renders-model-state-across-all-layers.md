---
type: source
title: "Observation: Harness --graph flag renders model state across all layers"
tags:
  - dialogue
  - harness
  - visualization
  - graph
  - engine-state
status: observation
created: 2026-09-08
updated: 2026-09-08
slug: obs-2026-09-08-harness-graph-flag-renders-model-state-across-all-layers
relevance: medium
observed_at: 2026-09-08T09:40:12.029Z
source_context: Extending the dialogue harness presentation with a model-state graph
---

# 🔍 Observation: Harness --graph flag renders model state across all layers

Extended src/dialogue/harness.py with a `--graph {ascii,dot,mermaid}` CLI flag (default off; also a `graph=` param on present(), so the curriculum path prints per-lesson). The graph renders end-of-run EngineState as a values×klines graph covering all four model layers: L=ltm, F=frame, W=work_list, R=refused. Core builder `_model_graph(state)` dedupes klines by (signature, nodes) accumulating layer glyphs, and maps each value to the layers it appears in (any role). Structure classes come from the semantic predicates only (unknown/id/canon/rel/under/over/misfit via is_unknown/is_identity/is_canon/is_connotation/classify_misfit) per the dialogue-dev discipline. Ask signatures get a "?" prefix via signifier.is_ask; values never heading a kline are listed under "-- referenced, never headed --" (the dangling-node signal trainers look for). Verified against mhall.ks (fresh + reloaded state), curricula/first-steps.md, -v hex labels, and -e structural supervisor.

*Relevance: medium*
*Context: Extending the dialogue harness presentation with a model-state graph*
*Tags: dialogue harness visualization graph engine-state*

---
*Observed: 2026-09-08T09:40:12.029Z*
