---
type: source
title: "Observation: Appendix added: the word-bit realisation in construction detail"
tags:
  - algebra
  - docs
  - appendix
  - realisation
status: observation
created: 2026-09-20
updated: 2026-09-20
slug: obs-2026-09-20-appendix-added-the-word-bit-realisation-in-construction-deta
relevance: high
observed_at: 2026-09-20T18:42:35.366Z
source_context: Adding word-bit realisation appendix to kalvin-algebra.md
---

# ⭐ Observation: Appendix added: the word-bit realisation in construction detail

Added "# Appendix — The word-bit realisation" to kalvin-algebra.md (after the Mary-had worked example): constructive detail for Def 2 populating the Def 1 value space via the 1-bit scheme. Sections: bit basis (one bit per distinct word, multi-subword word = one word one bit with subword ids OR'd into the token half, first-encountered assignment by ks TokenEncoder with cross-script seeding, compounds take no bit — OR-reduction of components, half-position 31 reserved for ASK, 32nd word = loud word-size-overflow error); layout ([ASK|word word|bpe token id] uint64, node = (word_bit << 32) | bpe_token_id, token half = inert encoding provenance masked at every content read, MASK = 0x7FFF_FFFF_0000_0000); capabilities table (∨ full-word OR, ∧ and ∖ masked, μ = popcount under mask, content equality = masked); laws check (bit-set facts, exact μ so Def 14 strict inequalities hold); marker discipline (ASK OR-ed at compile, no node carries it, signature never changes in derivation so it rides untouched, identity reads see it); address resolution (value is its own address, node references by holding signature — the two engine facts Def 2 names); costs (machine-word ops, depths accounted per word bit matching granularity-invariance). Def 2 now points to the appendix ("constructed in detail in the appendix"). All facts grounded in src/kalvin/signifier.py, src/ks/token_encoder.py, kline.ASK_SIG = 1<<63, significance.WORD_BITS. NOTE: an earlier parallel dispatch of edit + bash-append raced on the same file (edit's whole-file rewrite clobbered the append) — redone sequentially; never parallelize edits and appends to the same file.

*Relevance: high*
*Context: Adding word-bit realisation appendix to kalvin-algebra.md*
*Tags: algebra docs appendix realisation*

---
*Observed: 2026-09-20T18:42:35.366Z*
