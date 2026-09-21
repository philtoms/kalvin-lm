---
type: source
title: "Observation: KS/BPE audit: multi-token words correctly encoded; unicode charset undocumented"
tags:
  - kscript
  - ks
  - bpe
  - token-encoder
  - word-bits
  - encoding
  - audit
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-ks-bpe-audit-multi-token-words-correctly-encoded-unicode-cha
relevance: high
observed_at: 2026-09-21T11:54:13.552Z
source_context: Auditing KS/BPE cooperation for multi-subword exotic words
---

# ⭐ Observation: KS/BPE audit: multi-token words correctly encoded; unicode charset undocumented

KS/BPE cooperation audit for exotic words (hello?, don't, 3.14, café). Terminology settled: multi-token WORD is the real case (one KS word → multiple BPE subwords; hello?→[17203,63], Mary→[77,1524], café→4); multi-word TOKEN is structurally impossible (encode called per word); multi-word 'word' (M(ary had)) is a parse error since a083c47. Verified invariants: one word → one bit (first-encountered, 31 max, SystemError on 32nd); node = (bit<<32)|OR(subword_ids); values stable across compiles via shared word_bits; BPE never yields zero tokens for non-empty words (even ' - ... get ids) so no empty-OR edge; decode round-trips via labels. Two dimensions: upper-32 word bits are the SOLE content dimension (signifies/residual/measure/units all mask off BPE ids + ASK bit 63); lower-32 BPE OR is the identity/containment dimension (node_in uses full value — that's why the OR is carried). All-or-nothing word granularity: hello vs hello? no partial signification (superset lower-32 0x4333 vs 0x433f masked out). Compound sigs OR members' full values; repeated words collapse to one bit (MHALL probe = 4 bits for 5 chars, LL shared). FINDING: identifiers are unicode-wide — str.isalnum() admits café/naïve/CJK; coherent (word-list words were always unicode; identifier charset must ⊇ list-word charset for matching) but CONTEXT.md 'alphanumerics' reads ASCII — one-line doc clarification recommended. Verdict: nothing to fix; exotic symbols changed exercise frequency, not behavior.

*Relevance: high*
*Context: Auditing KS/BPE cooperation for multi-subword exotic words*
*Tags: kscript ks bpe token-encoder word-bits encoding audit*

---
*Observed: 2026-09-21T11:54:13.552Z*
