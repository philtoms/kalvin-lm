---
type: source
title: Split rewrite moves by preserved invariant to get termination for free
status: insight
category: design
created: 2026-09-10
updated: 2026-09-10
slug: content-preserving-vs-content-aligning-moves
---

# Split rewrite moves by preserved invariant to get termination for free

Kalvin's derivation relation has two move families with different invariants: witnessed moves (expand/contract against held canons) preserve σ(ν_A) exactly — granularity changes at constant content — while targeting moves (add/remove toward a target B) change content at fixed granularity, restricted to multiset difference sets. The invariants give the metatheory for free:

1. **Termination of content-aligning runs**: if each move strictly decreases the multiset symmetric difference D = |ν_A ∖ ν_B| + |ν_B ∖ ν_A| (each add/remove moves one value's count toward the target's count), any run terminates in ≤ D₀ steps — no monotone-potential assumptions needed, and the bound is computable. Add/remove "oscillation" is then not merely discouraged but unlicensable: re-adding a removed value requires it to be in the target's difference set, which shrinks monotonically.
2. **The cycling hazard is the content-*preserving* family**: σ-preserving moves keep every coarse measure (band, graded content) constant, so expand↔contract loops against the same witness are invisible to any significance-level potential. Termination of mixed runs is a strategy property (bound witnessed-move runs), not a theorem.

Lesson: when auditing "derivations terminate" claims, split moves by which observable they preserve — the family that preserves the measure can cycle undetected, and the family that changes it usually carries its own Lyapunov function in the license conditions. Check claimed counterexamples against the license rules first; here the doc's claimed hazard (add/remove oscillation) was impossible and the real one (witnessed-move loops) was unmentioned. Context: [[sources/obs-2026-09-10-layer-2-review-termination-claims-inverted-one-step-relation]], applied in [[sources/obs-2026-09-10-ks2-md-layer-2-written-m-b-witnessed-targeting-families-t1-t2]].

*Category: design*

---
*Captured: 2026-09-10*

## Related

_Add links to related pages._
