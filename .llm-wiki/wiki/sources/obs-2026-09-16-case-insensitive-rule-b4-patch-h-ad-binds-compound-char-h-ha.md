---
type: source
title: "Observation: Case-insensitive Rule B4 patch: h(ad) binds compound char H→had"
tags:
  - kscript
  - word-binding
  - patch
  - bugfix
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-case-insensitive-rule-b4-patch-h-ad-binds-compound-char-h-ha
relevance: high
observed_at: 2026-09-16T14:35:54.161Z
source_context: wdmh H binding fix
---

# ⭐ Observation: Case-insensitive Rule B4 patch: h(ad) binds compound char H→had

Fixed the H→"have" mis-binding in wdmh-underfit.ks: root cause was _patch_parent_canonicalise (ks/ast_emitter.py, Rule B4) matching the sig char against the parent kline chars CASE-SENSITIVELY via str.find — the authored witness h(ad) (lowercase) never matched MHALL's uppercase H, so the parent-MTS patch no-oped and H fell to the annotation binding (H→"have", initial of 'have' in '(what did Mary have)'). The rest of the binding system was already case-insensitive (bind_override stores char.lower(); resolve lowercases). Fix: find on .lower() of both sides — the letter binds, not the case. No script edit needed: the original h(ad) now patches MHALL's MTS to MHALL:[Mary, had, A, L, L] — 'had' the lowercase corpus word, the SAME atom (bit 0x21) the mhall state holds, so cross-script word continuity is preserved (H(ad) would have minted a new 'Had' atom — _extract_inline_word preserves sig-char case). The question's own WDMH MTS keeps H→have (its MTS expanded before the witness registers — the question's wording is correct). γ(WDMH, MHALL) now = shared {Mary} only = 1/7 ≈ 0.143 → byte 0x25 → S3, matching the user's "covers MHALL over M". CONTEXT.md Word Binding entry updated: 'The letter binds, not its case'. 51 tests pass; mhall compile byte-identical.

*Relevance: high*
*Context: wdmh H binding fix*
*Tags: kscript word-binding patch bugfix*

---
*Observed: 2026-09-16T14:35:54.161Z*
