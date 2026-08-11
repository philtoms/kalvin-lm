---
type: source
title: "Observation: EngineState four-store model: grounded→ltm, STM disconnected"
tags:
  - engine
  - ltm
  - stm
  - dialogue
  - engine-state
  - refactor
status: observation
created: 2026-08-11
updated: 2026-08-11
slug: obs-2026-08-11-enginestate-four-store-model-grounded-ltm-stm-disconnected
relevance: high
observed_at: 2026-08-11T16:32:17.306Z
source_context: "Lean harness: correcting the EngineState four-store memory model"
---

# ⭐ Observation: EngineState four-store model: grounded→ltm, STM disconnected

Corrected the EngineState memory model per the user's spec. EngineState now holds four stores that mirror the original kalvin memory tiers:

- `work_list` — the cogitator queue (incoming entries + ungrounded sigs/nodes unpacked from them). Independent of STM.
- `ltm` — ratified klines (renamed from `grounded`; the old name conflated the store with the S1-realising action).
- `frame` — outgoing kline proposals and identity requests.
- `stm` — Short-Term Memory, reserved for the expansion strategies' EXCLUSIVE use. Not wired into any logic.

Key corrections to the previous turn's (over-reaching) STM introduction:
1. STM is NOT a write-cascade index in the lean engine. Every STM call site from the previous turn was removed: `add_work`/`remove_work_at`/`note_ratified`/`frame_kline`/`unframe` no longer touch STM; the orphan-check helper, `has_seen`, and `find_by_nodes` were deleted; `from_dict` no longer re-indexes STM. STM is constructed empty in `__post_init__` and stays empty (verified: 0 entries after a full mhall run).
2. `work_list` and STM are maintained independently — work-list writes do not cascade anywhere.
3. `grounded` → `ltm` everywhere in the lean engine: field, `note_grounded`→`note_ratified`, `is_grounded`→`is_in_ltm`, `grounded_nodes`→`ltm_nodes`, persistence key (`to_dict`/`from_dict` now use "ltm"). The `_ground`/`_promote` engine method names stay (they describe the S1-realising action per CONTEXT's Grounding term, not the store).

Scope: `RationaliserState` in `dialogue/rationalise.py` (the verbatim reference) was NOT touched — it still has `grounded`. The rename is scoped to `EngineState`.

Verification: mhall trace byte-identical before/after (default and -v); all 1220 tests pass; STM empty after run; persistence round-trips `ltm` key with no STM in snapshot.

Files: src/dialogue/engine_state.py (rewritten), engine.py, expand_fit.py, similar_fit.py, harness.py, CONTEXT.md (STM glossary reverted to general form), docs/behaviour-notes.md (four-store model replaces the false cascade rule).

*Relevance: high*
*Context: Lean harness: correcting the EngineState four-store memory model*
*Tags: engine ltm stm dialogue engine-state refactor*

---
*Observed: 2026-08-11T16:32:17.306Z*
