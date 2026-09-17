---
type: source
title: "Observation: Hops tiered into EngineState.stm; flattened memory removed"
tags:
  - engine-state
  - stm
  - hop
  - refactor
status: observation
created: 2026-09-17
updated: 2026-09-17
slug: obs-2026-09-17-hops-tiered-into-enginestate-stm-flattened-memory-removed
relevance: high
observed_at: 2026-09-17T14:09:56.744Z
source_context: Tiering hop memory into EngineState
---

# ⭐ Observation: Hops tiered into EngineState.stm; flattened memory removed

Hops no longer use a flattened memory. EngineState gained `stm: list[KLine]` (working memory, hop writes), `add_stm` (add_work behaviour — dedup by signature+nodes), `extend_stm`, and `where(predicate, include_stm=False)` spanning STM as the last tier. `kalvin.hop`: `candidate_goals`/`trawl` take state and read `state.where(not is_terminal, include_stm=True)`; `Hop` holds `state`, writes `self.state.extend_stm(r.composed)`; `run_hops` takes state (reservoir removed; EngineState typed via TYPE_CHECKING to keep kalvin→dialogue runtime-independent). Engine passes `self._state` to run_hops; `Engine._writes` and `Engine._held` deleted. stm is not persisted (to_dict/from_dict untouched — "empty at session start", matching the old engine-level _writes). CONTEXT.md needed no change: Progressive Path/STM glossary already specified this; source caught up. Tests updated to build EngineState fixtures (tests/test_hop.py, test_derivation.py, test_ask_marker.py). Seven dev probes updated to the new seams (probe_live_scope, probe_derivation_monitor, probe_wdmh_scope, probe_ask_sig_blast, probe_hop_wdmh, probe_wdmh_goals, probe_scope_values).

*Relevance: high*
*Context: Tiering hop memory into EngineState*
*Tags: engine-state stm hop refactor*

---
*Observed: 2026-09-17T14:09:56.744Z*
