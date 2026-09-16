---
type: source
title: "Observation: Kalvin algebra translated to verified Maude template"
tags:
  - maude
  - formal-spec
  - docs
  - kalvin-algebra
status: observation
created: 2026-09-15
updated: 2026-09-15
slug: obs-2026-09-15-kalvin-algebra-translated-to-verified-maude-template
relevance: high
observed_at: 2026-09-15T16:47:10.098Z
source_context: Translating docs/kalvin-algebra.md to a Maude template
---

# ⭐ Observation: Kalvin algebra translated to verified Maude template

Translated docs/kalvin-algebra.md into an executable Maude specification template at docs/kalvin-algebra.maude (477 lines, 13 modules mapping spec §1–§13). Verified end-to-end with Maude 3.5.1 (downloaded from github.com/maude-lang/Maude releases, macos-arm64; no brew formula). All 9 fit shapes, σ evaluation, Δ₀=4, γ=1/8 evaluate correctly; searches reproduce §9 exactly: slot walk W0 finds [a,l,l] via crossover at o; Q0 alone reaches NO done state (w⇉[o] swaps misfit atoms — mismatch stays 4, correctly barred by the strict-decrease licence, so the composed correspondence is genuinely required); Q1 (M0 + w:[a,l,l]) reaches done in 3 value-equal witnesses ([a,l,l,h,m], [all,h,m], full contraction [mhall]); MARY-STUCK closes on [w,h,m]; appendix fragment stuck at entry, goal-side walk reaches anchor m, adoption done. Key Maude encoding decisions: values as AC sets with deduplicating union \/ (plain assoc-comm-id, no idem attr — avoids nonlinear AC patterns); node sequences as free monoid via juxtaposition `__` with assoc+id:nil; klines as `_:_`; memory as AC set with LHS decomposition `(N : NuK) , M` picking evidence; 4 rule schemas = one replace operation × 2 directions × 2 licences (witnessed canon expand/contract; targeting fwd/rev with strict |σ(νA) Δ σ(νB)| decrease + Def-14 misfit-region clauses); slot walks as separate WalkState rules licensed by occurrence only. Gotchas hit: klines need parens inside memory `,_`; search-command variables need sort annotations at every occurrence (D:DerivState, including in such-that clauses).

*Relevance: high*
*Context: Translating docs/kalvin-algebra.md to a Maude template*
*Tags: maude formal-spec docs kalvin-algebra*

---
*Observed: 2026-09-15T16:47:10.098Z*
