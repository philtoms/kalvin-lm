# ADR 0004: NLP Bindings Derived from KScript Comments

**Date:** 2026-06-09  
**Status:** Superseded by BindingScope inline resolution (spec v2.0, 2026-06-10)

> **Updated (KB-169, KB-174):** The two-pass "binding resolver" architecture described below has been superseded by `BindingScope` (`src/kscript/binding_scope.py`), which provides a simplified single-pass scope stack for inline resolution during AST emission. The `BindingResolver` and `NLPSymbolTable` modules have been removed (KB-174). The core binding semantics — first-letter matching, lexical scoping, occurrence counters — remain unchanged.

## Context

When Kalvin switches to the NLP tokenizer, compiled klines need to carry grammatically meaningful signatures (POS + DEP + MORPH) rather than character-bit signatures. Currently, `M` encoded under NLP-BPE gets the NLP type of the BPE token for the letter "M" — uninformative.

The KScript language already has a convention of using comments as semantic annotations:

```
MHALL == SVO =>
   S(ubject) = M
   V(erb) = H
```

These comments are currently discarded by the parser. They contain the exact information needed to bind single-character signatures to meaningful NLP words.

## Decision

Repurpose KScript comments as NLP word lists. The binding resolver (a new compilation pass) interprets comments as positional word sources for signature binding:

1. **Inline**: `S(ubject)` binds S to "Subject"
2. **Upward traversal**: unbound signatures resolve via enclosing scope
3. **Downward traversal**: unbound signatures resolve via subscript scope

Bindings follow standard lexical scoping — inner blocks can shadow outer bindings.

No source language changes. The lexer, token types, and grammar are unchanged. The parser is modified only to preserve comments as AST nodes instead of discarding them.

Unbound signatures fall back to Mod32 encoding, producing mixed NLP/Mod32 klines.

## Alternatives Considered

### New syntax for NLP word lists
Introduce a distinct syntactic form (e.g., `[Mary had a little lamb]` or `|Mary had a little lamb|`) to separate NLP word lists from throwaway comments.

**Rejected for MVP**: adds lexer complexity, fragments the language, and the mismatch-and-discard behavior (comments whose word count doesn't match any signature) provides sufficient disambiguation. Can be introduced later if confusion arises.

### Per-entry metadata on CompiledEntry
Store the identifier→NLP-word mapping directly in each compiled entry for decompiler recovery.

**Deferred**: a source map approach is cleaner but the design is premature. Current diagnostic needs are served by BPE-decoding nodes and the Trainer's LLM agent.

## Consequences

- Comments are no longer purely throwaway — they carry compilation semantics under NLP mode.
- Mod32 compilation is completely unaffected (binding resolver is skipped).
- Decompilation of NLP-compiled klines cannot recover original KScript identifiers from signatures alone (signatures are OR'd NLP types, losing identity). The Trainer's LLM agent and node BPE decodability provide diagnostic value instead.
- Mixed NLP/Mod32 klines within the same graph will require rationalisation to handle.

## Supersession (2026-06-10)

The separate resolution pass and positional word-list matching described in this ADR have been superseded by **BindingScope inline resolution** (spec `kscript-nlp-binding` v2.0). Key changes:

- The separate resolution pass has been eliminated. The ASTEmitter now resolves bindings inline during its single AST walk via a `BindingScope`.
- Positional-zip matching (word count must equal character count) has been replaced by **first-letter matching** (case-sensitive: `word[0] == char`) with an **occurrence counter** for disambiguation.
- Upward/downward traversal mechanisms have been replaced by a scope stack walk: characters seek from the current (innermost) scope first, then parent scopes upward.
- The mapping artefact (symbol table) has been replaced by `BindingScope`, a lightweight scope stack with no separate artefact.
- The inline binding mechanism is retained, enhanced with Rule 4 override patching of the parent kline's MCS CANONIZE entry.

See `specs/kscript-nlp-binding.md` v2.0 for the current architecture.
