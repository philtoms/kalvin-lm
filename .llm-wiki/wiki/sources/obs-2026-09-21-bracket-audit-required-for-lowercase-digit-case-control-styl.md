---
type: source
title: "Observation: Bracket audit: required for lowercase/digit/case-control, stylistic otherwise"
tags:
  - kscript
  - ks
  - annotations
  - syntax
  - design
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-bracket-audit-required-for-lowercase-digit-case-control-styl
relevance: low
observed_at: 2026-09-21T09:54:53.237Z
source_context: Auditing bracket purpose after the case-rule implementation
---

# 📝 Observation: Bracket audit: required for lowercase/digit/case-control, stylistic otherwise

Post-implementation audit of inline bracket purpose. Still required for: (1) lowercase-initial witnesses (h(ad) — no bracketless form by design; wdmh.ks depends on it), (2) digit tails (M(2) = word M2; bracketless M2 is ALL-UPPER → compound, emits extra MTS canon M2:[M,2] — different compilation), (3) tail case control (S(UBJECT) → 'SUBJECT'; bracketless yields as-written tails only). Optional/stylistic: uppercase-initial expansions — M(ood) ≡ Mood byte-identically; bracket is a visible 'this binds' marker only. Oddity: multi-word tails M(ary had) → word 'Mary had' (brackets can, identifiers can't contain spaces; design question). Residual gap: no designed form for a Capitalized literal (no expansion, no binding) — accidental escape is a trailing inert annotation (Subject(x) stays literal per the suppression guard). Author rule of thumb: write the word as you mean it; bracket when the word's case can't say what you mean.

*Relevance: low*
*Context: Auditing bracket purpose after the case-rule implementation*
*Tags: kscript ks annotations syntax design*

---
*Observed: 2026-09-21T09:54:53.237Z*
