---
type: source
title: "Observation: ks TokenEncoder now packs a word word: one bit per word, size 31"
tags:
  - ks
  - tokenizer
  - bpe
  - word-bits
status: observation
created: 2026-09-01
updated: 2026-09-01
slug: obs-2026-09-01-ks-tokenencoder-now-packs-a-word-word-one-bit-per-word-size-
relevance: high
observed_at: 2026-09-01T10:43:50.164Z
source_context: ks TokenEncoder word-bit scheme implementation
---

# ⭐ Observation: ks TokenEncoder now packs a word word: one bit per word, size 31

TokenEncoder rewritten to the word-word scheme: node = (word_bit << 32) | bpe_id. One bit per distinct word, first-encountered, bits 0-30 (WORD_SIZE=31); the 32nd distinct word raises SystemError (word size overflow). Multi-subword words (Mary) are one bit with subword ids OR-reduced — compound-word decomposition machinery (\_emit_mts_for_tokens) removed. Multi-char uppercase sigs not registered by MTS (sigless-annotation initials like WWW...) compose their sig from nodes via signature_of instead of taking a bit; MTS CANONICALZES defs still register in \_compound_sigs and block-canon refs reuse it (MHALL = 5 word bits, no bit of its own). ASK_BPE_TOKEN moved 1<<31 → 1<<63 (word-word bit 31) in signifier.py; signifies/residual now mean word-bit overlap. All `from kalvin.nlp_tokenizer import BPETokenizer` imports (16 sites) repointed to kalvin.bpe_tokenizer. CONTEXT.md updated: Token ID, Terminal (two shapes — compound-word form retired), ASK. Pre-existing unrelated breakage: kalvin.expand imports is_s1 from significance which no longer has it.

_Relevance: high_
_Context: ks TokenEncoder word-bit scheme implementation_
_Tags: ks tokenizer bpe word-bits_

---

_Observed: 2026-09-01T10:43:50.164Z_
