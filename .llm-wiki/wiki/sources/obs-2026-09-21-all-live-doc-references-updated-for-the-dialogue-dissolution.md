---
type: source
title: "Observation: All live doc references updated for the dialogue dissolution"
tags:
  - docs
  - refactor
  - verification
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-all-live-doc-references-updated-for-the-dialogue-dissolution
relevance: medium
observed_at: 2026-09-21T15:21:47.550Z
source_context: Updating document references after src/dialogue refactor
---

# 🔍 Observation: All live doc references updated for the dialogue dissolution

Document-reference sweep after the src/dialogue dissolution completed. This turn: README.md kalvin/ tree made fully accurate (added derivation.py, hop.py, bpe_tokenizer.py, mod_tokenizer.py alongside the new engine.py/engine_state.py entries; dev/dialogue tree was already in); live wiki pages fixed — entities/k-engine.md (`dialogue.harness` → `dev.dialogue.harness`), entities/harness.md (launch command → `PYTHONPATH=src:. python -m dev.dialogue.harness`). Previously done: CONTEXT.md (3 refs), .pi/skills/dialogue-dev/SKILL.md, wiki concepts/trainer.md + entities/harness.md path refs. Verified zero live references remain via perl lookbehind sweep over git-tracked files (excluding .llm-wiki raw/meta/sources history, which is immutable by design; .discoveries/gaps.json is auto-regenerated). Only remaining old-style mention in code is trace_full.py's dead `from dialogue.actors import` (historical, module never existed in main tree). Tests 95/95.

*Relevance: medium*
*Context: Updating document references after src/dialogue refactor*
*Tags: docs refactor verification*

---
*Observed: 2026-09-21T15:21:47.550Z*
