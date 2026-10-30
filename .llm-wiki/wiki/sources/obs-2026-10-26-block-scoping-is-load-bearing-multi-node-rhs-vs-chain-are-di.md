---
type: source
title: "Observation: Block scoping is load-bearing; multi-node RHS vs chain are distinct structures"
tags:
  - ks
  - kscript
  - block-scoping
  - chaining
  - review
status: observation
created: 2026-10-26
updated: 2026-10-26
slug: obs-2026-10-26-block-scoping-is-load-bearing-multi-node-rhs-vs-chain-are-di
relevance: high
observed_at: 2026-10-26T17:45:04.722Z
source_context: Reviewing data/ks.md truth-doc gaps; user pushed back that scaffolding implies chaining
---

# ⭐ Observation: Block scoping is load-bearing; multi-node RHS vs chain are distinct structures

Ran ASTEmitter directly (Lexer→Parser→ASTEmitter+BindingScope) on mhall.ks vs a flattened variant (same statements, no indented blocks). As written: SVO canon = SVO:[Subject,Verb,Object] (chars resolved via desugared S(ubject)/V(erb)/O(bject) inline bindings pre-registered in the pushed `=>` subscript frame before expansion, ast_emitter.py:302), DENOTES at scope 2, ALL:[a,little,lamb] with the two Ls disambiguated to little/lamb by counter-reset-on-push (binding_scope.py:36). Flattened: SVO:[S,V,O] raw chars, DENOTES at scope 0 — same text, different indentation, different klines, no error. Also: `A < B C` compiles to ONE connotation BCA:[A] (multi-node RCONNOTES) while `A < B < C` compiles to TWO (BA:[A], CB:[B]) with B dual-role (node of outer, sig of inner) — the operator token is the only discriminator; data/ks.md's scaffolding line names the recursion but gives no rule to distinguish multi-node RHS from chaining, nor innermost-block attachment. Refines the earlier 14-gap review: the chaining gap is specifically the missing shape-row/discriminator, not the recursion concept.

*Relevance: high*
*Context: Reviewing data/ks.md truth-doc gaps; user pushed back that scaffolding implies chaining*
*Tags: ks kscript block-scoping chaining review*

---
*Observed: 2026-10-26T17:45:04.722Z*
