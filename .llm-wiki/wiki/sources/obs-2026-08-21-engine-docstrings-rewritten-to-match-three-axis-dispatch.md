---
type: source
title: "Observation: Engine docstrings rewritten to match three-axis dispatch"
tags:
  - dialogue
  - engine
  - docs
status: observation
created: 2026-08-21
updated: 2026-08-21
slug: obs-2026-08-21-engine-docstrings-rewritten-to-match-three-axis-dispatch
relevance: medium
observed_at: 2026-08-21T13:48:07.528Z
source_context: Documentation drift pass over dialogue engine
---

# 🔍 Observation: Engine docstrings rewritten to match three-axis dispatch

route/_fast_route/_slow_route/cogitate docstrings in src/dialogue/engine.py described an older structural-dispatch model while code dispatched on three axes (stamped sig, structural sig, question-vs-statement). Rewrote docstrings around the actual axes, deleted commented-out countersign block in cogitate, made _fast_route return None explicitly. Result: _countersignature_proposals/_operand_pairings/_pairing_resolved are now uncalled dead code (their only caller was the comment). Separately: tests/test_dialogue_smoke.py fails to import (missing ScriptTrainee from dialogue) — pre-existing, reproduced on clean tree.

*Relevance: medium*
*Context: Documentation drift pass over dialogue engine*
*Tags: dialogue engine docs*

---
*Observed: 2026-08-21T13:48:07.528Z*
