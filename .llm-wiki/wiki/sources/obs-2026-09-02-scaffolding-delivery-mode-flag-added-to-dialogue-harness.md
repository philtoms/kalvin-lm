---
type: source
title: "Observation: Scaffolding delivery-mode flag added to dialogue harness"
tags:
  - harness
  - scaffolding
  - dialogue
status: observation
created: 2026-09-02
updated: 2026-09-02
slug: obs-2026-09-02-scaffolding-delivery-mode-flag-added-to-dialogue-harness
relevance: high
observed_at: 2026-09-02T12:19:33.141Z
source_context: Adding scaffolding delivery-mode flag to dialogue harness
---

# ⭐ Observation: Scaffolding delivery-mode flag added to dialogue harness

Harness now supports two scaffolding delivery modes behind `--scaffolding batch|on-demand` (default batch, the current behaviour). `Harness.__init__` takes `scaffolding: Literal["batch","on-demand"]="batch"`, threaded through make_engine/load_engine and the CLI. batch feeds the whole annotation group with the opener; on-demand feeds only the opener (batch_sources = [opener]) and the group's entries answer K's asks via `_answer`. Word-identity ride-along at S1 follows the batch sources, so on-demand withholds scaffold word identities too. CONTEXT.md Scaffolding term updated. Pre-existing: ruff I001 import order and mypy bus.py errors in src/training/harness/bus.py.

*Relevance: high*
*Context: Adding scaffolding delivery-mode flag to dialogue harness*
*Tags: harness scaffolding dialogue*

---
*Observed: 2026-09-02T12:19:33.141Z*
