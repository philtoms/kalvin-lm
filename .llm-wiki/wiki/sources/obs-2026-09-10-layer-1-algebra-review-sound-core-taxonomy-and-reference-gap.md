---
type: source
title: "Observation: Layer-1 algebra review: sound core, taxonomy and reference gaps"
tags:
  - formalisation
  - algebra
  - docs
  - review
status: observation
created: 2026-09-10
updated: 2026-09-10
slug: obs-2026-09-10-layer-1-algebra-review-sound-core-taxonomy-and-reference-gap
relevance: high
observed_at: 2026-09-10T12:29:17.738Z
source_context: Reviewing layer 1 (the algebra) of docs/kalvin-symbolic.md
---

# ⭐ Observation: Layer-1 algebra review: sound core, taxonomy and reference gaps

Reviewed layer 1 of docs/kalvin-symbolic.md. Core (two sorts V/K, σ as order/multiplicity-forgetting join, one fit classifier F(v,ν) serving both kline fit and pairwise relationship) is sound. Defects found: (1) fit classification is two independent axes — value-level gap/excess vs node-level coverage — but presented as one flat list, so species overlap (Connotation is simultaneously No-fit and under+over; abc:[abd] is an unnamed covered single-node under+over); needs an explicit decision tree. (2) The band map shape→S1..S4 exists only implicitly in the table; it is band = S4 if ν=[], S1 if v=σ(ν), else S2 if any node covered, S3 otherwise — should be stated as a Definition in §1. (3) Reference structure ("node is signature of another kline") is named but never defined, and the "memory is a DAG of witnesses" claim is false: a:[a] self-loops, reciprocal connotation p:[q]/q:[p] is a 2-cycle, and xy:[xy,x] is a self-referential canon. (4) The ask atom in A is dangling — never used in layer 1; should be an annotation, not an atom. (5) "Memory is a set of decomposition witnesses" overclaims — held misfits are not witnesses; reword as claims (closed canon / open misfit). (6) "Section (right inverse)" language is misapplied per-kline; canon is a local splitting, a section is a global choice. (7) Def 6's "special case" is not C(A,A) (always canon); single-kline fit is F(s,ν) directly, and significance of (A,B) deliberately ignores A's signature — worth stating. (8) Minor: ¬/1 undeclared in the Boolean signature, finiteness forward-ref points to §2.5 not §2.3, ∧/¬/C are operations despite "only ∨ and kline formation" claim, head/slots vs signature/nodes dual naming.

*Relevance: high*
*Context: Reviewing layer 1 (the algebra) of docs/kalvin-symbolic.md*
*Tags: formalisation algebra docs review*

---
*Observed: 2026-09-10T12:29:17.738Z*
