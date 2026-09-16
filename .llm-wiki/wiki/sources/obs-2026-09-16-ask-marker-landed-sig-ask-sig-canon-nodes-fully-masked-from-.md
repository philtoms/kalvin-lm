---
type: source
title: "Observation: ASK marker landed: sig|ASK_SIG:[canon nodes], fully masked from measurement"
tags:
  - ask
  - compiler
  - engine
  - hop
  - harness
  - masking
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-ask-marker-landed-sig-ask-sig-canon-nodes-fully-masked-from-
relevance: critical
observed_at: 2026-09-16T15:43:52.061Z
source_context: Landing the ASK marker + masking + is_ask self-exclusion
---

# 🔴 Observation: ASK marker landed: sig|ASK_SIG:[canon nodes], fully masked from measurement

Implemented the ASK marker design: `==`/annotation/bare asks compile to `sig|ASK_SIG:[canon nodes]` (e.g. `WDMH|ASK_SIG:[what,did,Mary,have]`). ASK_SIG = 1<<63 (word-word bit 31) defined in kalvin/kline.py with `is_ask(sig)`; TokenEncoder ORs it after compound-registry reuse; ast_emitter marks every ASK emission is_ask=True and the COUNTERSIGNS ask now carries the MTS canon's nodes (mts_idx passed to _emit_operator_entries). Masking is total per the user's must-have decision: kline._TYPE_MASK and significance.WORD_BITS both exclude the bit (0x7FFF_FFFF_0000_0000) so signifies/residual/word_atom_count/misfit_mass/gamma never see it — no manufactured gap (is_exact True, underfit/overfit False, J identical to clean form). Structural gates: sig_level returns S4 for ask-marked klines; is_canon excludes asks (question ≠ answer). Def 22 self-exclusion gained the is_ask step: a question is never a goal, and an ask never heads its own goal list via its canon (base-sig comparison — verified: after canon release the top goal is had:[did,have], not the canon). Engine._propose masks the bit from proposal signatures (the marker belongs to the question, not the answer) so supervisor `under()` lookups key correctly. Harness: goals keyed by masked base; graded() grades ask-marked entries at γ; is_ask_content() withholds the canon (non-ask, noded, base in goals); _drive's ask-vs-proposal discriminator is is_ask; _answer's ask branch releases pools under the masked base. Decoder ASK constructs the marked canon-noded form (script-declared nodes still rejected). Docs: CONTEXT.md Token ID + ASK entries, algebra §13 table + marker note, script-reading.md. Fixed stale conftest tokenizer import (nlp_tokenizer→bpe_tokenizer). New tests/test_ask_marker.py (5 tests); 56/56 pass. Verified end-to-end: wdmh-underfit ask feeds once at γ-graded 0x25 (S3 band, no refusal), canon never leaks, 0 escalations; mhall unchanged (J(MHALL,SVO)=0 honestly — goal atoms are the abstract role words). Probe dev/dialogue/probe_ask_sig_blast.py. Uncommitted (per AGENTS.md). Known stale debt: scripts/dialogue-*.json + tests/_fixtures use removed ops — need regeneration.

*Relevance: critical*
*Context: Landing the ASK marker + masking + is_ask self-exclusion*
*Tags: ask compiler engine hop harness masking*

---
*Observed: 2026-09-16T15:43:52.061Z*
