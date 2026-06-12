# Rename UNSIGNED → IDENTITY

## Spec References

- @specs/kscript.md §6.2 (significance level assignment), §7.1 (identity operator), §8 (MCS expansion)
- @specs/signature.md (signature value semantics)
- @specs/countersign-resolution.md (significance levels by operator)
- @specs/agent.md §3 (grounding assessment)
- @specs/stm.md (sig == 0 behaviour)
- @specs/nlp-curriculum-compat.md (decomposition entries)

## Context

The operator for bare-node klines (empty nodes, no relationships) was called UNSIGNED. The domain term is IDENTITY — it's the structural endpoint where rationalisation stops. See CONTEXT.md glossary entry "Identity".

## Implementation Tasks

### Task 1: Rename in `ks/` source

**Scope:** `src/ks/ast_emitter.py`, `src/ks/token_encoder.py`

Replace every `"UNSIGNED"` string literal with `"IDENTITY"`:
- `ast_emitter.py`: `_op_to_str()` default return, `_emit_entry()` calls, docstrings, comments
- `token_encoder.py`: `_SIG_LEVELS` dict key, `CompiledEntry` default `op`, MCS emission `op=`, docstrings

Acceptance: `grep -rn '"UNSIGNED"' src/ks/` returns nothing.

### Task 2: Rename in `kalvin/` runtime source

**Scope:** `src/kalvin/agent.py`, `src/kalvin/expand.py`

Search for `"UNSIGNED"` or references to "unsigned" as an op name. Update comments that say "unsigned" in the operator sense.

Acceptance: No operator-name references to "unsigned" remain in `src/kalvin/`.

### Task 3: Rename in tests

**Scope:** All test files that reference `"UNSIGNED"` as an op string.

Mechanical rename: `"UNSIGNED"` → `"IDENTITY"` in every assertion, constructor, and filter.

Test files affected:
- `tests/test_ks_ast_emitter.py`
- `tests/test_ks_token_encoder.py`
- `tests/test_ks_compiler.py`
- `tests/test_ks.py`

Acceptance: `grep -rn '"UNSIGNED"' tests/` returns nothing. All affected tests pass.

### Task 4: Rename in scripts

**Scope:** `scripts/kalvin_test.py`, `scripts/ks_test.py`

Any op-string references to `"UNSIGNED"`.

Acceptance: `grep -rn '"UNSIGNED"' scripts/` returns nothing.

## Design Decisions

1. **AST `op=None` unchanged.** `op=None` in the parser/AST means "no operator written". The emitter maps this to `"IDENTITY"`. No new `TokenType.IDENTITY` token. Rationale: adding a token for something the user didn't write is misleading.

2. **Old `kscript/` left untouched.** It uses `IDENTITY` at S1 for a different purpose. The collision is temporary and confined to a package being replaced. Rationale: `kscript/` is frozen pending decompiler migration.

## Status

- [x] Specs updated
- [x] Plan created
- [ ] Task 1 (ks/ source)
- [ ] Task 2 (kalvin/ runtime)
- [ ] Task 3 (tests)
- [ ] Task 4 (scripts)
