---
type: source
title: "Observation: Symbol range committed (4d1bbb9): caseless words + MTS alnum guard"
tags:
  - kscript
  - ks
  - lexer
  - syntax
  - implementation
  - committed
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-symbol-range-committed-4d1bbb9-caseless-words-mts-alnum-guar
relevance: high
observed_at: 2026-09-21T10:16:13.087Z
source_context: Committing the symbol-range extension
---

# ⭐ Observation: Symbol range committed (4d1bbb9): caseless words + MTS alnum guard

Committed 4d1bbb9 (symbol range) on dialogue, after f653367 (case rule). Identifiers now admit `- . _ '` beyond alphanumerics via lexer allowlist `_IDENT_PUNCT` (frozenset in src/ks/lexer.py); caseless-first (digit/punct) = literal word — the existing case predicates already route caseless chars (isupper False, expansion False, gate blocks attraction) so zero parser/emitter changes were needed beyond the MTS alnum guard. Defect found during investigation and fixed: `'A-1'.isupper()` is True (uncased chars ignored) so MTS char-decomposed punctuation — `_emit_mts` now requires `sig.isalnum()`, punctured identifiers are words. Prose-word matching enabled: kline-position identifiers can equal word-list words (`(don't stop)` / `D = don't` → IDENTITY don't:[don't] via attraction + literal match + self-denote collapse). Held back as reserved-syntax candidates: `? ! ,`. Known accepted cost: lone `-` between spaces lexes as an identifier (a - b = three signatures) instead of erroring. 98 tests green, mhall/wdmh fixtures byte-identical.

*Relevance: high*
*Context: Committing the symbol-range extension*
*Tags: kscript ks lexer syntax implementation committed*

---
*Observed: 2026-09-21T10:16:13.087Z*
