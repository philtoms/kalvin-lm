---
type: source
title: "Observation: Word Binding three-tier precedence settled"
tags:
  - kscript
  - word-binding
status: observation
created: 2026-08-17
updated: 2026-08-17
slug: obs-2026-08-17-word-binding-three-tier-precedence-settled
relevance: high
observed_at: 2026-08-17T17:02:37.602Z
source_context: Refining KScript binding scope rules
---

# ⭐ Observation: Word Binding three-tier precedence settled

Word Binding refined to three tiers: (1) inline annotations override all others and additionally bind uppercase chars in the immediate parent scope only (not beyond); (2) prefix annotations bind by scope, inner before outer; (3) resolved bindings — a file-level char→word memory of earlier successful resolutions — bind after all annotations, so the trailing O in wdmh_connot.ks's `W > O` binds to Object. Two fixes in src/ks: binding_scope.py gained the resolved map and parent-scope override propagation; ast_emitter.py makes MTS char expansion non-consuming on occurrence counters (counters_snapshot/restore) and threads canon operand resolutions to child scope sigs via per-char queues (_node_res_q), so duplicate chars (the two Ls in ALL) resolve to little and lamb exactly once each. CONTEXT.md Word Binding glossary updated. Tests updated/added in test_ks_binding_scope.py and TestDuplicateCharOccurrences in test_ks_compiler.py; 1221 pass.

*Relevance: high*
*Context: Refining KScript binding scope rules*
*Tags: kscript word-binding*

---
*Observed: 2026-08-17T17:02:37.602Z*
