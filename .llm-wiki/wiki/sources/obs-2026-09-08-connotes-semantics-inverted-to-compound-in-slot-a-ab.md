---
type: source
title: "Observation: CONNOTES semantics inverted to compound-in-slot (A:[AB])"
tags:
  - kscript
  - compiler
  - connotation
  - semantics
  - refactor
status: observation
created: 2026-09-08
updated: 2026-09-08
slug: obs-2026-09-08-connotes-semantics-inverted-to-compound-in-slot-a-ab
relevance: high
observed_at: 2026-09-08T09:38:28.284Z
---

# ⭐ Observation: CONNOTES semantics inverted to compound-in-slot (A:[AB])

KScript CONNOTES semantics changed per user spec: `A > B` now emits `A:[AB]` (was `AB:[B]`) — the compound sits in the signature's slot instead of heading the kline. RCONNOTES `A < B` emits `B:[BA]`, preserving the invariant that `A < B` compiles identically to `B > A`. Self-reference still collapses to IDENTITY. Consequence: connotation klines flip from underfit to overfit classification (residual bits now on the node side), and cogitator.connotateY edges now run sig→compound (A→AB) instead of both-participants→genus. Changed: src/ks/ast_emitter.py (emission + new SymbolicEntry.concat field), src/ks/token_encoder.py (removed sig-side concat_head recovery; new _compose_concat composes the compound NODE from component word values, registered in _compound_sigs, never takes a word bit), src/dialogue/engine_state.py (EngineState.is_connotation containment mirrored to node_in(signature, nodes[0]) — old check node_in(nodes[0], signature) would reject every new-scheme connotation kline), CONTEXT.md (4 glossary spots). Tests were removed from the repo Sep 1 (commit c31c1ce), so verification was via inline scripts. Pre-existing breakage untouched: sig_in called in dialogue/reentry.py + expand_fit.py but defined nowhere; 5 pre-existing mypy errors.

*Relevance: high*
*Tags: kscript compiler connotation semantics refactor*

---
*Observed: 2026-09-08T09:38:28.284Z*
