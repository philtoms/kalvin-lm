---
type: source
title: "Observation: EngineState.stm renamed back to work_list"
tags:
  - dialogue
  - engine-state
  - refactor
status: observation
created: 2026-09-03
updated: 2026-09-03
slug: obs-2026-09-03-enginestate-stm-renamed-back-to-work-list
relevance: high
observed_at: 2026-09-03T11:28:02.817Z
source_context: Renaming EngineState.stm to work_list to make room for kalvin.stm import
---

# ⭐ Observation: EngineState.stm renamed back to work_list

Renamed EngineState.stm to work_list — the list of entries fed to Kalvin via the slow route — freeing the name `stm` for a future `kalvin.stm.STM` import into state. Methods renamed add_stm→add_work, remove_stm→remove_work, remove_stm_at→remove_work_at. Callers updated in src/dialogue/engine.py, src/dialogue/harness.py, dev/dialogue/probe_rationalise.py. Persistence key changed to "work_list" (no saved snapshots existed). CONTEXT.md STM glossary untouched — it refers to the model concept (kalvin.stm), not EngineState. This reverses the 2026-08-17 rename.

*Relevance: high*
*Context: Renaming EngineState.stm to work_list to make room for kalvin.stm import*
*Tags: dialogue engine-state refactor*

---
*Observed: 2026-09-03T11:28:02.817Z*
