---
type: source
title: "Observation: ASK klines now sig|ASK_BPE_TOKEN with original canonical signature"
tags:
  - ks
  - compiler
  - ask
status: observation
created: 2026-08-27
updated: 2026-08-27
slug: obs-2026-08-27-ask-klines-now-sig-ask-nlp-token-with-original-canonical-sig
relevance: high
observed_at: 2026-08-27T13:03:47.689Z
source_context: Reworking ASK kline compilation to signature-agnostic ask bit
---

# ⭐ Observation: ASK klines now sig|ASK_BPE_TOKEN with original canonical signature

ASK klines no longer use the registered "ASK" token as signature. SymbolicEntry gained `is_ask`; asks keep their original canonical signature — the compound itself for bare compounds (ABC → ABC|ASK_BPE_TOKEN:[A,B,C]), or the annotation's word initials for sigless annotations ((a big cat) → ABC|ASK_BPE_TOKEN:[a big cat]). New constant ASK_BPE_TOKEN = 1<<31 in src/kalvin/nlp_tokenizer.py (type-word bit; dictionary allocates bits 0–29). TokenEncoder ORs it into sig_uint64 when entry.is_ask. Files: src/ks/ast_emitter.py (\_emit_ask, bare-compound path, \_emit_entry), src/ks/token_encoder.py, src/kalvin/nlp_tokenizer.py, CONTEXT.md ASK glossary entry, tests/test_ks.py, tests/test_ks_ast_emitter.py. A sigless ask's canonical signature matches the equivalent authored compound ask's signature exactly. All ks tests pass (205).

_Relevance: high_
_Context: Reworking ASK kline compilation to signature-agnostic ask bit_
_Tags: ks compiler ask_

---

_Observed: 2026-08-27T13:03:47.689Z_
