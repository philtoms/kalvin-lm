---
type: source
title: "Observation: EngineState renamed to Memory (kalvin.memory)"
tags:
  - rename
  - refactor
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-enginestate-renamed-to-memory-kalvin-memory
relevance: high
observed_at: 2026-09-21T15:37:33.930Z
source_context: Renaming EngineState to Memory after the dialogue dissolution
---

# ⭐ Observation: EngineState renamed to Memory (kalvin.memory)

Renamed src/kalvin/engine_state.py → src/kalvin/memory.py and class EngineState → Memory (uncommitted). Mechanical sed across engine.py, hop.py (TYPE_CHECKING), harness.py, build_state.py, 2 probes, 5 test files; README tree entry updated; 4 live wiki pages (k-engine, stm, ltm, frame) `EngineState` → `Memory`. Import-order (I001) regressions in engine.py, test_derivation.py, test_hop.py re-sorted — lesson: renames that change alphabetical position require re-sorting import blocks in every importer. Zero EngineState/engine_state references remain in code or docs. 95/95 tests, ruff at 252 baseline.

*Relevance: high*
*Context: Renaming EngineState to Memory after the dialogue dissolution*
*Tags: rename refactor*

---
*Observed: 2026-09-21T15:37:33.930Z*
