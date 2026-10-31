---
type: source
title: "Observation: data/ks.md review: orientation doc, not truth doc — 14 gaps"
tags:
  - docs
  - ks
  - kscript
  - review
  - truth-doc
status: observation
created: 2026-10-26
updated: 2026-10-26
slug: obs-2026-10-26-data-ks-md-review-orientation-doc-not-truth-doc-14-gaps
relevance: high
observed_at: 2026-10-26T15:28:53.378Z
source_context: Reviewing data/ks.md for usefulness as a truth doc
---

# ⭐ Observation: data/ks.md review: orientation doc, not truth doc — 14 gaps

Reviewed data/ks.md (57 lines, orphaned — nothing references it) as a candidate truth doc for the KScript DSL. Strengths: five-level reading (surface/symbolic/structure/ask-goal/scaffolding), structure table, decode-back sanity check. Gaps: (1) features used by wdmh.ks or implemented in src/ks but absent: import, word expansion desugaring (Capitalized → M(ood), parser.py:88 — rewrites the doc's own mhall example: Subject < M compiles as S(ubject) < M), compound signatures (_expand_compound; mhall = ask initials of its annotation per _emit_ask), binding rules B1/B4 (binding_scope.py; L = Mod then L = O in mhall is unexplained rebinding), INDENT/DEDENT scope rules, chained operators (Object < Query < ALL =>), identifier grammar + reserved chars, bare annotation statement (what Object). (2) Wrong claim: "surface semantics do not survive compilation" contradicted by _emit_ask (ask sig = word initials, nodes = words), _feed_annotation (bindings), inline overrides. (3) ask→goal asserted but == SVO never specified (echoes ks2 G5); a:[] unknown purpose unstated despite _emit_terminal_unknowns/_emit_identity_if_needed. (4) No truth-doc apparatus: no normative status/scope/version (src says KScript v3), no compiled-output inventory (KValue/SymbolicEntry, implicit identity entries), no pipeline placement (lexer→parser→emitter→tokenizer-32768→harness), no error behaviour. Verdict: good orientation doc, not a truth doc.

*Relevance: high*
*Context: Reviewing data/ks.md for usefulness as a truth doc*
*Tags: docs ks kscript review truth-doc*

---
*Observed: 2026-10-26T15:28:53.378Z*
