---
type: concept
title: Underfit and Overfit
created: 2026-09-14
updated: 2026-09-14
---

# Underfit and Overfit

The two misfit quantities — one notion read on the two parties of a kline.

## Definition

For a pair (s, ν) (Def 9, `docs/kalvin-algebra.md`):

```text
u = s ∧ ¬σ(ν)    # underfit
o = σ(ν) ∧ ¬s    # overfit
```

The **underfit** u contains atoms claimed by the head but not supplied by the
nodes. The **overfit** o contains atoms the nodes carry beyond the signature.
Both zero → exact; both nonzero → the Under+over shape. Underfit and Overfit
also name two of the nine fit [[concepts/structural-significance|shapes]]
(single-node instances: Denotation is single-node underfit).

Renamed from gap/excess on 2026-09-14 throughout the algebra doc and the
CONTEXT.md glossary. `src/` and `dev/algebra` still use gap/excess as local
variable names.

## The asymmetry

Underfit is positively located in ν_A (each underfit atom sits on a node — a
diagnostic **slot**); overfit is positively located in ν_B — "absence has no
location on A". Consequences:

- Slot derivation (Def 17) seeds walks from underfit slots; **overfit
  relationships have no slots** on their own side — the slot criterion is
  vacuous when u = ∅, stranding the derivation in Stuck case 2 ([[concepts/ask]]).
- An overfit slot of C(A,B) is an underfit slot of C(B,A) — reverse reads are
  half-sanctioned by the mirror clause (Def 13) and direction-free klines (§9).

Application of overfit evidence already works (selection + adopt-fwd); only
construction (the slot walk) is missing.

## Links

- [[concepts/misfit]] — the structure these quantities measure
- [[concepts/fit]] / [[concepts/structural-significance]] — what the quantities feed
- [[concepts/s2-expansion]] — S2 expansion driven by underfit
- [[sources/obs-2026-09-14-gap-excess-renamed-to-underfit-overfit-in-algebra-doc-and-gl]] — the rename
- [[sources/obs-2026-09-14-slot-derivation-gap-overfit-relationships-have-no-slots]] — the asymmetry and design fork

_Avoid_: gap / excess (superseded names), content gap (reworded to content mismatch).
