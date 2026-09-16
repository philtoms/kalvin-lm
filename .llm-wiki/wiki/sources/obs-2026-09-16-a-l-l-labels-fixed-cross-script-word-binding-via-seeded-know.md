---
type: source
title: "Observation: A/L/L labels fixed: cross-script word binding via seeded known_words"
tags:
  - compiler
  - word-binding
  - labels
  - drift
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-a-l-l-labels-fixed-cross-script-word-binding-via-seeded-know
relevance: high
observed_at: 2026-09-16T17:02:51.290Z
source_context: Fixing A/L/L labels in the wdmh compile
---

# ⭐ Observation: A/L/L labels fixed: cross-script word binding via seeded known_words

Investigated and fixed the A/L/L labels confusion: in the wdmh-underfit compile, MHALL's MTS nodes showed as chars 'A','L','L' instead of the words a/little/lamb. Root cause: char→word binding is resolved from the script's own word lists (first-letter + occurrence counter) plus inline witnesses — mhall.ks binds A→a, L→little, L→lamb from its corpus line, but the underfit question script's word list (what did Mary have) has no such words, and the char→word memory (BindingScope.resolved_bindings) is file-level — the harness passed only word→bit continuity (word_bits), never bindings. The chars minted fresh (A=0x...41, L=0x...4c) — different VALUES from the state's a/little/lamb words, hence both the confusing labels and the long-standing MHALL value drift between scripts. Fix: compile_source/Compiler gained known_words (the prior state's words in acquisition order = word_bits insertion order = corpus order), seeded as the compile's outermost root word list; load_engine sets harness.known_words = list(state.word_bits). Tier-2 mechanics then bind A→a, L→little, L→lamb (occurrence counter handles the repeated L) while the script's own lists stay most-recent-first ahead of the seed. Result: MHALL:[Mary, had, a, little, lamb] with sig 0x1f00001ffd = the mhall state's exact value — the drift is GONE (the canon grounds on arrival); ALL:[a,little,lamb] = the corpus canon value; the == goal entry is clean ([had, ALL] — the what pollution also disappeared with the seed); the answer proposal now reads WDMH:[a, little, lamb, had, Mary]. CONTEXT.md Word Binding documents the cross-script seed. tests/test_known_words_seed.py (unseeded mints chars; seeded binds words + value continuity); 61/61 green. Known side effect: scaffold steps now emit junk proposals (a:[Object], little:[a, Mod, lamb, a]) — with unified values the scaffold klines' re-entry chains find routes through the state's role denotations; the S4 stub declines them. Same noise class as the pre-existing MHALL:[MHALL] identity proposal.

*Relevance: high*
*Context: Fixing A/L/L labels in the wdmh compile*
*Tags: compiler word-binding labels drift*

---
*Observed: 2026-09-16T17:02:51.290Z*
