---
type: source
title: "Observation: Wiki corrected for EngineState four-store model"
tags:
  - wiki
  - engine
  - ltm
  - stm
  - frame
status: observation
created: 2026-08-11
updated: 2026-08-11
slug: obs-2026-08-11-wiki-corrected-for-enginestate-four-store-model
relevance: medium
observed_at: 2026-08-11T16:36:43.772Z
source_context: Fixing the LLM wiki after the EngineState grounded→ltm / STM-disconnect refactor
---

# 🔍 Observation: Wiki corrected for EngineState four-store model

Corrected the LLM wiki after the EngineState four-store refactor left it stale. The previous turn's source packets (obs-2026-08-11-stm-introduced-into-enginestate-as-write-cascade-index and stm-cascade-removals-orphan-check) described an STM write-cascade that was removed the same day. Rather than delete them (they're accurate event records), added ⚠️ supersession banners at the top of each pointing to the four-store correction packet.

Updated canonical pages:
- entities/stm-short-term-memory.md — split into production-Model role (write cascade, every write reaches it) vs lean EngineState role (reserved for expansion, currently unwired). Front-matter description updated.
- entities/k-engine.md — added the four-store model (work_list/ltm/frame/stm) as the central EngineState fact.
- entities/ltm-long-term-memory.md — noted the lean `EngineState.ltm` field name and the rename from `grounded`.
- concepts/frame.md — split into production-Model Frame (monotonic) vs lean EngineState `frame` (emission memory, NOT monotonic — unframe removes consumed asks). The old "Frame is monotonic" claim was misleading for the lean engine.

Did NOT change: concepts/grounding.md, entities/isgroundable.md, entities/promote.md (they use "grounded/groundable" in the concept sense, accurate per CONTEXT's Grounding term; engine method names `_ground`/`_promote` didn't change); concepts/ratify.md (inter-agent dialogue concept; the engine's `note_ratified` method borrows the term for the internal LTM-write, documented in the four-store packet).

Verified: 0 gaps, 0 new orphans, mhall trace still byte-identical, 19 relevant tests pass.

*Relevance: medium*
*Context: Fixing the LLM wiki after the EngineState grounded→ltm / STM-disconnect refactor*
*Tags: wiki engine ltm stm frame*

---
*Observed: 2026-08-11T16:36:43.772Z*
