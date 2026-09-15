---
type: source
title: "Observation: CONNOTES/DENOTES shapes swapped — connotes is the compound, denotes the plain"
tags:
  - kscript
  - compiler
  - connotation
  - denotation
  - semantics
status: observation
created: 2026-09-15
updated: 2026-09-15
slug: obs-2026-09-15-connotes-denotes-shapes-swapped-connotes-is-the-compound-den
relevance: high
observed_at: 2026-09-15T10:07:32.859Z
source_context: Swapping CONNOTES/DENOTES compiled structures in the kscript compiler
---

# ⭐ Observation: CONNOTES/DENOTES shapes swapped — connotes is the compound, denotes the plain

Compiler shapes swapped per user spec: `>` CONNOTES now emits the compound AB:[B] (concat=[sig,*nodes]); `<` RCONNOTES emits BA:[A] (concat=[*nodes,sig], reading order, A<B ≡ B>A value-identical); `=` DENOTES emits plain A:[B]. Derived consequence recorded in glossary: `==` COUNTERSIGNS (A:[B]+B:[A]) is now a pairwise denotes — model.is_countersigned accordingly ratifies mutual `=` pairs to S1. Structural predicates swapped in src/kalvin/kline.py: is_connotation = case 6 covered gap-only AB:[B] (S2), is_denotation = case 4 uncovered A:[B] (S3). band_significance stamps swapped: CONNOTES→S2, DENOTES→S3 (COUNTERSIGNS stays S2). S3-bridge machinery renamed to follow the plain shape's new name: cogitator.connotate→denotate, reentry._connotate→_denotate, expand_fit._connotations→_denotations, pivot_fill._crossover_connotations→_crossover_denotations, expand/proposals s3_connotations→s3_denotations. Lexer/parser/token symbol↔name bindings UNCHANGED (unlike branch `dialogue` 8d3e9d2, which did the inverse: renamed symbols, kept shapes). Also updated: engine_state (is_connotation→is_denotation method), harness -e help, llm_supervisor + curriculum_generator prompts, CONTEXT.md (Relational Tokens, Shape, nine-structure table), README (Connote S3→S2), docs/kalvin-algebra.md (case 4/6 species names, worked examples, § reciprocal Denotation pairs countersigned), dialogue-dev skill references, tests (species swap), dev probes (renamed call sites). Verified: 44 tests pass, mypy error count unchanged (138 pre-existing), engine runs all data/scripts/*.ks, compiled shapes/stamps/word-bits verified inline. NOTE: branch `dialogue` (8d3e9d2, one commit ahead of HEAD) contains the conflicting inverse variant and will conflict on merge.

*Relevance: high*
*Context: Swapping CONNOTES/DENOTES compiled structures in the kscript compiler*
*Tags: kscript compiler connotation denotation semantics*

---
*Observed: 2026-09-15T10:07:32.859Z*
