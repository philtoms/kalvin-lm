---
type: source
title: "Observation: Compound words no longer take word bits"
tags:
  - kscript
  - token-encoder
  - word-bits
status: observation
created: 2026-09-03
updated: 2026-09-03
slug: obs-2026-09-03-compound-words-no-longer-take-word-bits
relevance: high
observed_at: 2026-09-03T10:53:47.409Z
source_context: Excluding compound words from compiled word_bits
---

# ⭐ Observation: Compound words no longer take word bits

TokenEncoder no longer assigns word bits to compound words: CONNOTES concatenation sigs (SubjectMary = Subject+Mary, Verbhad, ObjectQuery, QueryALL) now follow the compound rule — signature = OR of component word values (entry nodes + the head word recovered by stripping the node concat from the sig string, prefix or suffix), registered in _compound_sigs/_compound_labels for reuse and display. mhall word_bits now hold only real words (Mary..Mod). Also typed _encode_word -> KNode (returns KNode(0, word) for empty) and node_values: list[KNode], fixing pre-existing mypy signature_of arg-type error. Persisted states' word_bits may still contain old compound entries (not migrated). F401 SIG_S1 unused import is pre-existing.

*Relevance: high*
*Context: Excluding compound words from compiled word_bits*
*Tags: kscript token-encoder word-bits*

---
*Observed: 2026-09-03T10:53:47.409Z*
