---
type: source
title: "Observation: UNKNOWN op unified into ASK; ask bit deleted"
tags:
  - ks
  - compiler
  - ask
  - unknown
  - ask-bit
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-unknown-op-unified-into-ask-ask-bit-deleted
relevance: high
observed_at: 2026-09-16T13:13:09.431Z
source_context: Unifying UNKNOWN into ASK and dropping the ask bit
---

# ⭐ Observation: UNKNOWN op unified into ASK; ask bit deleted

UNKNOWN op removed, unified into ASK; ASK_BPE_TOKEN bit (bit 63 / word-word bit 31) deleted entirely. Rationale (user): ASK and UNKNOWN filled the same semantic space — same structure (`A:[]`), same significance (S4) — the dedicated bit was no longer required (matches the 2026-09-10 observation "ask is structural S4 only"). Changes: ast_emitter `_op_to_str(None)`→"ASK", all UNKNOWN emissions→ASK, SymbolicEntry.is_ask field removed; token_encoder ask-bit step removed (ask signature == the plain compound/word signature — a `==` ask now shares its signature with the MTS canon, making harness heads-pool lookups unify); signifier ASK_BPE_TOKEN + KSignifier.is_ask removed; KDbg.op default "" (empty=unset) replacing "UNKNOWN" sentinel; significance/_OP_TO_SIG, decoder DIALOGUE_OPS (UNKNOWN→ASK), kline _OP_SYMBOLS pruned; harness model-graph "?=ask" display dropped. Docs: CONTEXT.md (Token ID bit 31 unused; Relational Tokens single ASK entry), algebra §13 "declared an ask", script-reading.md table. 51 tests pass; all three data/scripts compile+run. Remaining stale: tests/_fixtures MHALL_TURNS + scripts/dialogue-*.json use UNKNOWN/COUNTERSIGNS ops (unused, decoder would reject — needs deliberate regeneration).

*Relevance: high*
*Context: Unifying UNKNOWN into ASK and dropping the ask bit*
*Tags: ks compiler ask unknown ask-bit*

---
*Observed: 2026-09-16T13:13:09.431Z*
