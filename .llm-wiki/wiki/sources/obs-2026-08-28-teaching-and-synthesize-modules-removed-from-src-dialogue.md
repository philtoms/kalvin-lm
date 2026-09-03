---
type: source
title: "Observation: Teaching and synthesize modules removed from src/dialogue"
tags:
  - dialogue
  - cleanup
status: observation
created: 2026-08-28
updated: 2026-08-28
slug: obs-2026-08-28-teaching-and-synthesize-modules-removed-from-src-dialogue
relevance: medium
observed_at: 2026-08-28T15:21:52.189Z
source_context: Removing Teaching and synthesize from src/dialogue
---

# 🔍 Observation: Teaching and synthesize modules removed from src/dialogue

Deleted src/dialogue/teaching.py and src/dialogue/synthesize.py (already excluded from engine). Cleaned up: unused Teaching import in engine_state.py; the -t/--training CLI flag and its Engine.TRAINING setter block in harness.py (flag no longer existed on Engine); stale actor exports (ScriptTrainee, ScriptTrainer, Rationalising*) and actor/runner docstring in dialogue/__init__.py; deleted tests/test_dialogue_smoke.py which imported removed actors and was already failing. Remaining test failures (runner, tokenizer, cogitator, adapter) are pre-existing and unrelated.

*Relevance: medium*
*Context: Removing Teaching and synthesize from src/dialogue*
*Tags: dialogue cleanup*

---
*Observed: 2026-08-28T15:21:52.189Z*
