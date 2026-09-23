---
type: source
title: "Observation: Engine/rationaliser identity swap done (module names + classes)"
tags:
  - architecture
  - refactor
  - naming
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-engine-rationaliser-identity-swap-done-module-names-classes
relevance: high
observed_at: 2026-09-21T16:02:51.631Z
source_context: Engine/rationaliser module rename refactor
---

# ⭐ Observation: Engine/rationaliser identity swap done (module names + classes)

Refactor step 1 of the engine/rationaliser split, on branch `dialogue` (staged, uncommitted): src/kalvin/engine.py (the one-shot kline rationaliser, class Engine, "the rationalising engine") → src/kalvin/rationaliser.py with class Rationaliser; src/kalvin/rationaliser.py (the orchestrator pipeline, class Rationaliser) → src/kalvin/engine.py with class Engine, protocol RationaliserAdapter → EngineAdapter, alias Agent = Engine. Blast radius updated: cogitator/events/__init__/memory/model/expand docstrings+imports, training harness (adapter.py EngineAdapter + _EngineLike, __main__ registered bus participant class "Rationaliser"→"Engine" matching training.harness.yaml `class: Engine`), trainer, TUI supervisors, scripts (kalvin_test, encode_text), tests/test_ratification.py, all dev/dialogue probes (make_engine/load_engine → make_rationaliser/load_rationaliser, h.engine → h.rationaliser, eng_mod → rat_mod), and .pi/skills/dialogue-dev vocabulary. 95 tests pass; mhall.ks dialogue run verified. Step 2 pending: link the new Engine to the new Rationaliser (currently unlinked).

*Relevance: high*
*Context: Engine/rationaliser module rename refactor*
*Tags: architecture refactor naming*

---
*Observed: 2026-09-21T16:02:51.631Z*
