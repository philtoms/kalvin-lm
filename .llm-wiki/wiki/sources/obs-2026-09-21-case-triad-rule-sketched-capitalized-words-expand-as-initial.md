---
type: source
title: "Observation: Case-triad rule sketched: Capitalized words expand as Initial(tail)"
tags:
  - kscript
  - ks
  - parser
  - word-binding
  - annotations
  - syntax
  - design
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-case-triad-rule-sketched-capitalized-words-expand-as-initial
relevance: high
observed_at: 2026-09-21T09:13:18.853Z
source_context: Sketching the case-triad disambiguation rule for optional bracket syntax
---

# ⭐ Observation: Case-triad rule sketched: Capitalized words expand as Initial(tail)

Sketched the case-triad disambiguation rule for bracketless KS annotations: ALL-UPPER multi-char = MTS compound (existing `sig.isupper()` rule), Capitalized (initial upper + ≥1 lower) = word expansion reading as Initial(tail) e.g. Mood ≡ M(ood), lowercase-first/single-char = literal (brackets stay the escape hatch: h(ad)). Implementation: parser-level normalization at Signature construction (two hooks in parser.py — `_parse_operator_scope` sig side, `_parse_items` item side), guarded by "explicit annotation suppresses" (Mood(x) stays literal); emitter/lexer untouched so equivalence with the bracketed path is by construction. Empirically verified on the Mary script vs its bracketed equivalent: 13/15 klines byte-identical; deltas are the binding side effects — lamb:[O]→lamb:[Object] (bound initial resolves later bare O) and SVO:[S,V,O]→SVO:[Subject,V,Object] (Rule B4 patch; V stays raw — partial patching is visible). Inherited asymmetry kept deliberately: sig-side fill-if-empty, node-side unconditional. Doctrine-aligned with CONTEXT.md:226 ("the letter binds, not its case; the word's case is the word"). Doc home: CONTEXT.md Annotation/Word Binding entries; kalvin-algebra.md §13/§14 untouched per §14 delegation. Open knob: whether lowercase-first words should also expand (recommended no).

*Relevance: high*
*Context: Sketching the case-triad disambiguation rule for optional bracket syntax*
*Tags: kscript ks parser word-binding annotations syntax design*

---
*Observed: 2026-09-21T09:13:18.853Z*
