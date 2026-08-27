---
type: source
title: "Observation: Compiler emits ASK klines for bare compounds and sigless annotations"
tags:
  - kscript
  - compiler
  - asks
status: observation
created: 2026-08-27
updated: 2026-08-27
slug: obs-2026-08-27-compiler-emits-ask-klines-for-bare-compounds-and-sigless-ann
relevance: high
observed_at: 2026-08-27T11:00:24.417Z
---

# ⭐ Observation: Compiler emits ASK klines for bare compounds and sigless annotations

Compiler ask output implemented (src/ks/ast_emitter.py, src/kalvin/significance.py): (1) a bare compound scope (no operation, e.g. `WDMH`) keeps the MTS canon + identities but the canon entry is replaced to sig="ASK", op="ASK" and removed from the MTS dedup registry (else a later authored canon for the same compound is silently swallowed); (2) a sigless annotation (not immediately followed by an OperatorScope, incl. EOF or another annotation) emits ASK:[words] via new `_emit_ask`. `band_significance("ASK") → SIG_S4`. ASK encodes as a plain registered token via the encoder default path (no compound-registry involvement, so multiple asks coexist). Updated tests in tests/test_ks.py + tests/test_ks_ast_emitter.py; CONTEXT.md KScript relational-tokens glossary gained the ASK entry. Suite back to pre-existing baseline (5 failed / 10 errors, all adapter/import).

*Relevance: high*
*Tags: kscript compiler asks*

---
*Observed: 2026-08-27T11:00:24.417Z*
