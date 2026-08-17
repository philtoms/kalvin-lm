---
type: source
title: "Observation: Model/Memory definitions settled: tiers are relations, not storage"
tags:
  - model
  - memory
  - frame
  - stm
  - ltm
  - grounding
  - glossary
  - domain-model
status: observation
created: 2026-08-17
updated: 2026-08-17
slug: obs-2026-08-17-model-memory-definitions-settled-tiers-are-relations-not-sto
relevance: high
observed_at: 2026-08-17T11:16:52.277Z
source_context: Working up Model/Memory glossary definitions after user discarded working-tree changes
---

# ⭐ Observation: Model/Memory definitions settled: tiers are relations, not storage

Settled the Model/Memory definitions with the user and applied them after the user discarded all uncommitted working-directory changes (the 08-14 Frame/LTM glossary rewrite and its wiki edits are gone; repo is at commit 5ca030b).

The key correction to my earlier proposal: memory is NOT a substrate beneath the model — it is part of the model, because the tiers are defined by how Kalvin relates to held klines DURING rationalisation, not by where klines are stored. Kalvin traversing the model is conscious of (rationalises): what it was just thinking about (STM = recent attention, written by attention itself); what the focus is and how it's shifting (Frame = focus relation); what it actually considers grounded knowledge (Frame AND LTM = commitment relation). Same kline content, different epistemic relation per tier. These relations only exist for something that rationalises — hence memory cannot be outside the model. Tier changes (promotion, framing, eviction) are rationalisation work, not storage bookkeeping. Untiered klines in a file are a serialisation, not memory.

Applied:
- CONTEXT.md: new **Model** and **Memory** glossary entries (in Rationalisation section, before Frame); rewrote Frame/STM/LTM entries to the relation framing. LTM's discarded promotion story not restored (left open).
- Wiki: created concepts/model and concepts/memory; rewrote entities/stm-short-term-memory, concepts/frame, entities/ltm-long-term-memory, concepts/grounding to the relation framing, all linking up to model/memory as parent concepts.

Consequences flagged: EngineState.stm being empty/unwired is now "a missing mode of self-awareness", not a reserved index; STM wiring should be attention-driven (whatever cogitation touches hits STM), not a write cascade. Also noted: the "grounded nothing/8-groundings" state I observed earlier was caused by an Aug-12 engine refactor bug in _ground's cascade (grounds/observes `kline` instead of `entry`; cogitate lost its _is_groundable arm; direct state.ground appends no observation) — that refactor is now DISCARDED from the working tree, but it was committed at f40a77a..5ca030b (amended). Verify engine state before further tuning.

*Relevance: high*
*Context: Working up Model/Memory glossary definitions after user discarded working-tree changes*
*Tags: model memory frame stm ltm grounding glossary domain-model*

---
*Observed: 2026-08-17T11:16:52.277Z*
