---
type: source
title: "Observation: CONNOTES/DENOTES structural semantics inverted and made compound"
tags:
  - kscript
  - compiler
  - semantics
status: observation
created: 2026-08-28
updated: 2026-08-28
slug: obs-2026-08-28-connotes-denotes-structural-semantics-inverted-and-made-comp
relevance: high
observed_at: 2026-08-28T14:09:14.777Z
source_context: Changing KScript conote/denote structural semantics
---

# ⭐ Observation: CONNOTES/DENOTES structural semantics inverted and made compound

KScript operator semantics changed per user spec: CONNOTES `A > B` now emits `{AB:[B]}` (A is a kind of B; B connotes A) — compound signature = concatenation of raw sig + nodes, one entry per scope; RCONNOTES `A < B` emits `{AB:[A]}`; DENOTES `A = B` now emits `{A:[B]}` (A is a B; forward direction, was reversed). Self-reference on any of the three collapses to IDENTITY `{A:[A]}`. Multi-node: single aggregated entry (e.g. `A > B C` → `{ABC:[B,C]}`). Changed: src/ks/ast_emitter.py (_emit_operator_entries + removed DENOTES subscript-identity special-casing since DENOTES now heads its own sig), CONTEXT.md, llm_supervisor.py and curriculum_generator.py prompts, tests (test_ks.py, test_ks_ast_emitter.py, test_ks_compiler.py). Pre-existing unrelated breakage: ImportError `is_s1` from kalvin.significance affecting test_agent/test_adapter/expand tests.

*Relevance: high*
*Context: Changing KScript conote/denote structural semantics*
*Tags: kscript compiler semantics*

---
*Observed: 2026-08-28T14:09:14.777Z*
