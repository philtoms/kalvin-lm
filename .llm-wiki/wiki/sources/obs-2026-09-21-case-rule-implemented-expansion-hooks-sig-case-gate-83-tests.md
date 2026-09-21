---
type: source
title: "Observation: Case rule implemented: expansion hooks + sig-case gate, 83 tests green"
tags:
  - kscript
  - ks
  - parser
  - word-binding
  - binding-scope
  - implementation
  - tests
status: observation
created: 2026-09-21
updated: 2026-09-21
slug: obs-2026-09-21-case-rule-implemented-expansion-hooks-sig-case-gate-83-tests
relevance: high
observed_at: 2026-09-21T09:50:00.069Z
source_context: "Implementing the case rule: parser expansion + sig-case attraction gate"
---

# ⭐ Observation: Case rule implemented: expansion hooks + sig-case gate, 83 tests green

Implemented the case rule: (1) src/ks/parser.py — `_is_word_expansion` (Capitalized multi-char: first upper, ≥1 lower) + `_expand_word` (Mood → Signature('M') + inline '(ood)', the exact bracketed AST) with hooks in `_parse_operator_scope` (sig side) and `_parse_items` (item side), guarded by explicit-annotation-suppresses (Mood(x) stays literal). (2) src/ks/binding_scope.py — resolve() sig-case gate: lowercase chars skip ambient tiers (word lists + resolved memory) but explicit overrides remain visible to any case — this placement was chosen after tracing wdmh.ks: inside a CANONICALISES subscript the h(ad) override is registered by _resolve_nodes' node-side collection BEFORE the sig-side fill-if-empty check, so keeping overrides visible preserves that path exactly. (3) CONTEXT.md Annotation/Word Binding entries updated (bracketless reading + gated attraction + case-blindness scoped to authored bindings). (4) tests/test_ks_case_rule.py (13 tests). Behavior deltas beyond design: top-level authored lowercase witnesses now fire (h(ad) under (have) → had, was blocked inert — position independence); bare Capitalized word compiles to IDENTITY not ASK (self-carried word = known word); test_ask_marker's lowercase witness (cat)\nc updated to sig-case C with a complementary lowercase-stays-ask lock. Fixtures mhall.ks/wdmh.ks compile byte-identically (diffed). Suite: 83 passed (70 baseline). One test-expectation bug of mine found by run: bare uppercase A attracts the article 'a' → little:[a].

*Relevance: high*
*Context: Implementing the case rule: parser expansion + sig-case attraction gate*
*Tags: kscript ks parser word-binding binding-scope implementation tests*

---
*Observed: 2026-09-21T09:50:00.069Z*
