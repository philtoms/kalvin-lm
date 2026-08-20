---
type: source
title: "Observation: Similar-fit strategy removed; ExpandFit is the only S2 misfit strategy"
tags:
  - dialogue
  - refactor
  - s2-strategy
status: observation
created: 2026-08-20
updated: 2026-08-20
slug: obs-2026-08-20-similar-fit-strategy-removed-expandfit-is-the-only-s2-misfit
relevance: high
observed_at: 2026-08-20T15:17:35.135Z
source_context: Removing pluggable S2 fit strategies
---

# ⭐ Observation: Similar-fit strategy removed; ExpandFit is the only S2 misfit strategy

Removed the dynamic fit-policy loading in src/dialogue: deleted similar_fit.py and EngineState.similar_fit_candidates, removed the _STRATEGIES registry and the -s/--strategy CLI flag from harness.py, and hardcoded ExpandFit in make_engine/load_engine (misfit_cls parameter dropped). README tree updated. tests/test_dialogue_smoke.py import error (ScriptTrainee) is pre-existing and unrelated.

*Relevance: high*
*Context: Removing pluggable S2 fit strategies*
*Tags: dialogue refactor s2-strategy*

---
*Observed: 2026-08-20T15:17:35.135Z*
