---
type: entity
title: KScript
description: The DSL that authors training material by declaring klines and labelling them with Target Significance. A compiler/provenance concern — it produces structures and target labels, never a participant's lived significance.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# KScript

The language that authors training material.

## Overview

A KScript script declares [[concepts/kline|klines]] and, through its
[[concepts/relational-tokens]] and the structures they generate, labels each
with a [[concepts/target-significance]] — the answer a trainee must learn to
derive for itself. KScript is a compiler/provenance concern: it produces
structures and their target labels, never a participant's lived significance.

Key compiler concepts: [[concepts/word-binding]] (resolving single-character
signatures to words), [[concepts/mts-multi-token-signature|MTS]] (compound
signatures), and the five relational tokens.

In the codebase, KScript is a library — there is no standalone compiler CLI.
Compilation is invoked in-process via `ks.compiler.compile_source` by the
harnesses and by scripts (see the observation
`obs-2026-08-11-no-python-m-kscript-entrypoint-exists`). The compiler pipeline:
lexer → parser → ASTEmitter → binding scope → TokenEncoder → encoded KLines.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — the KScript section of the glossary
- [[concepts/relational-tokens]] — how scripts declare provenance
- [[concepts/word-binding]] — how signatures resolve to words
- [[concepts/mts-multi-token-signature]] — compound signatures
- [[concepts/target-significance]] — what each kline is labelled with
- [[entities/tokenencoder]] — the symbolic → encoded KLines stage
