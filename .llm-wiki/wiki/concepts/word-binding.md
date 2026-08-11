---
type: concept
title: Word Binding
description: The association of a single-character KScript signature with a word, resolved through annotations — fill-if-empty at top level, unconditional override inline.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# Word Binding

The association of a single-character KScript signature with a word.

## Definition

Bindings are resolved through annotations in the source and scoped by
[[concepts/relational-tokens|relational-token]] boundaries: a character
resolves to the most recent matching word in its scope. Two annotation kinds
bind with different strength:

- **top-level annotation** (on a scope signature) — binds only if the character
  is currently unbound. Fill-if-empty; never overrides an outer binding.
- **inline annotation** (on an item) — binds unconditionally, overriding any
  outer binding for that occurrence.

Each identity occurrence is bound exactly once by the most specific annotation
that applies to it, so one character never acquires two competing tokens.

Binding chooses the structure for a bare signature: a word-bound bare signature
compiles to an [[concepts/identity]] `{A:[A]}`, an unbound one to an
[[concepts/unknown]] `{A:[]}`. The structure then determines the
[[concepts/target-significance]].

_Avoid_: comment mapping (the binding is a specific compiler artefact, not a
general comment feature); rebind (a top-level annotation never overrides; an
inline annotation always does — use the specific kind).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[concepts/identity]] — what a word-bound bare signature compiles to
- [[concepts/unknown]] — what an unbound bare signature compiles to
- [[concepts/relational-tokens]] — token boundaries scope bindings
- [[entities/kscript]] — the language bindings live in
- [[entities/tokenencoder]] — where bindings are resolved to tokens
