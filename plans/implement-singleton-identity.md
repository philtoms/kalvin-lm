# Singleton Identity & Target Significance — Implementation Plan

**Spec:** @specs/kscript.md §6.2, §7.1, §7.3, §7.6, §8, §8.3, §11.3, §14.1, §14.5, §14.12
**Context:** @CONTEXT.md §Target Significance, §Relational Tokens, §Identity, §Unknown
**Status:** Done

## Goal

A bare single-character signature compiles to a self-referential **Identity**
`{S: [S]}` (S1) when it is **word-bound**, and to the empty **Unknown**
`{S: []}` (S4) only when genuinely orphaned. Self-denote (`A = A`) compiles to
the self-referential Identity `{A: [A]}` (S1), binding-independent. MTS
constituent characters follow the same uniform rule. The compiled significance
byte is reframed throughout as the **Target Significance** (the trainer's answer
key) — see @CONTEXT.md §Target Significance.

This generalizes the `IDENTITY` op that already exists in the TokenEncoder for
§11.3 compound-word identities (`{packed: [packed]}`) to singletons, self-denote,
and MTS components.

Word binding is the **sole** discriminator between Identity and Unknown for a
lexical-UNKNOWN singleton. MTS-component introduction and node-referencing do
not elevate an unbound singleton.

## Spec changes (done)

@specs/kscript.md rewritten:
- §6.2 retitled **Target Significance**; IDENTITY row added to the op→level table.
- §7.1 UNKNOWN token: binding-aware (bound → Identity `{S:[S]}`, unbound → Unknown `{S:[]}`).
- §7.3 self-denote: `A = A` → `{A: [A]}` Identity (no longer collapses).
- §7.6 subscript identity gap-filling: binding-aware (outcome-agnostic).
- §8 / §8.3 MTS component identities: binding-aware; dedup keys on resolved form.
- §14.1 (split unbound/bound), §14.5 (self-Identity), §14.12 (bound components flip to S1; S/V/O stay S4 per B4 scope discipline).
- Test matrix: KS-33 rewritten; KS-33a/b/c, KS-19a added; KS-19/KS-34/KS-38 wording updated.

## Tasks

### Task 1 — `IDENTITY` op in the significance map (`src/kalvin/significance.py`)

Add `"IDENTITY": SIG_S1` to `_OP_TO_SIG`. `band_significance("IDENTITY")` then
returns SIG_S1; `band_significance("UNKNOWN")` stays SIG_S4.

Spec: @specs/kscript.md §6.2 table.
Test mapping: KS-33b (bound singleton → S1), KS-33 (self-denote → S1).

### Task 2 — ASTEmitter emits IDENTITY for bound singletons & self-denote (`src/ks/ast_emitter.py`)

The emitter currently stamps `op="UNKNOWN"` with empty nodes for every bare
singleton and for self-denote, ignoring whether resolution produced a word.
Branch on resolution outcome:

- **Bare singleton** (`_process_scope`, op == "UNKNOWN", single-char sig,
  `mts_idx is None`): if `_resolve_inline_or_scope` returned a word different
  from the raw char → emit `op="IDENTITY"` with `nodes=[sig]`. Else keep
  `op="UNKNOWN"`, `nodes=[]`. (Multi-char sigs are unaffected — `_emit_mts`
  handles them and compounds cannot form an identity.)
- **Self-denote** (`_emit_operator_entries`, DENOTES branch, `node == sig`):
  emit `op="IDENTITY"` with `nodes=[sig]` always (binding-independent).
- **MTS component identities** (`_emit_mts`): replace the unconditional
  `UNKNOWN {char: []}` emission with the binding-aware rule — resolved word
  differs from raw char → `IDENTITY {word: [word]}`; else `UNKNOWN {char: []}`.
  Dedup (`_mts_identity_seen`) keys on the resolved form so a bound component
  dedups against a bare bound singleton of the same word.
- **Subscript gap-filling** (`_emit_identity_if_needed`): apply the same
  binding-aware branch (the resolved char is already computed at the call
  sites).

The `SymbolicEntry.op` docstring gains `IDENTITY` to the union.

Spec: §7.1, §7.3, §7.6, §8, §8.3.
Test mapping: KS-33/33a/33b/33c, KS-19/19a, KS-34.

### Task 3 — TokenEncoder builds `{S: [S]}` for IDENTITY (`src/ks/token_encoder.py`)

The main-entry path (`_encode_entries_for_entry`) currently builds the KLine
from `entry.nodes` (empty for UNKNOWN). For an `IDENTITY` op entry:

- `nodes` is already `[sig]` (symbolic) from Task 2; encode it normally. The
  resulting KLine is `{sig_uint64: [sig_uint64]}` — self-referential.
- The `IDENTITY` op flows into `band_significance(entry.op)` (Task 1) → SIG_S1.
- No change to the §11.3 compound-word path (`_emit_mts_for_tokens`) — it
  already emits `{packed: [packed]}` IDENTITY S1 and is the precedent.

Confirm the `is_compound_def` / packed-sig guard at the UNKNOWN branch still
holds: an IDENTITY entry's sig is a single token (never packed), so the
"`op == UNKNOWN` and `sig_is_packed`" skip does not apply.

Spec: §6.2, §11.3.
Test mapping: KS-33/33b, KS-37.

### Task 4 — Tests

Update existing assertions and add new ones:

- `tests/test_ks_compiler.py` `TestKS36WordBound.test_compiled_entries_valid`:
  the valid-op set gains `"IDENTITY"` (this also fixes the pre-existing failure
  from the COMPOUND_TOKEN migration, which already emits IDENTITY).
- `tests/test_ks.py` `test_ks36_word_bound_example`: "Mary" entries are now
  `IDENTITY` (or the resolved word is the signature of a self-ref kline), not
  `UNKNOWN`/`CANONIZES`. Re-assert against §14.12's revised table.
- `tests/test_ks_ast_emitter.py`: add cases for bound singleton →
  `SymbolicEntry(op="IDENTITY", nodes=[sig])`, self-denote → IDENTITY,
  MTS-bound component → IDENTITY, unbound cases unchanged.
- `tests/test_ks_token_encoder.py`: the 3 pre-existing failures
  (`TestMultiTokenMTS::*`, `TestDedupMTS::test_dedup_same_word_twice`) stem from
  the COMPOUND_TOKEN migration emitting IDENTITY; update expected ops/structures
  to the self-ref identity form.
- Add KS-33a/b/c, KS-19a coverage.

Test mapping: KS-33/33a/33b/33c, KS-19/19a, KS-34, KS-36, KS-37, KS-38.

## Design Decisions

| Decision                                                | Rationale                                                                                              |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `IDENTITY` op generalized (not a new sig path)          | The §11.3 compound-word identity already uses `IDENTITY` + `{S:[S]}` + SIG_S1. Singletons reuse it.   |
| Binding is the sole singleton discriminator             | Grill decision Q1. Ties Identity/Unknown to the existing §10 BPE mechanism.                            |
| Self-denote is binding-independent Identity             | Grill Q8a: once the author writes the self-ref, the structure is fixed at S1; binding no longer votes. |
| MTS components follow the singleton rule uniformly      | Grill Q9a: no MTS carve-out; dedup keys on resolved form.                                              |
| B4 patches canon nodes only, not component identities   | Grill Q12a: preserves `DH = h(ad)` scope-leak prevention. §14.12 S/V/O stay S4.                        |
| `band_significance` retained, not torn out              | Grill keystone: the byte is the Target Significance (answer key), not a structural derivation.         |

## Follow-on: aggregate-only decomposition

After the initial implementation, the B4-staleness problem surfaced: MTS
component identities emitted at expansion time could not be retroactively
patched by Rule B4 inline overrides in subscript scopes (the `DH = h(ad)`
scope-leak prevention forbids it). Rather than weaken B4, the resolution is
that **decomposition emits only the aggregate** — components are values
inside the aggregate, not headed klines. An author wanting a headed kline for
a component writes it as a bare singleton (§7.1).

This applies uniformly to both decomposition mechanisms:

- **MTS (§8):** emits only the CANONIZES canon `{compound: [chars]}`. No
  per-character component entries. Removed `_mts_identity_seen` and the
  component-emission loop from `_emit_mts`.
- **Compound-words (§11.3):** emits only the self-referential identity
  `{packed: [packed]}`. No per-subword component entries. Removed `_decomposed`
  from the TokenEncoder; added `_compound_identity_emitted` to dedup the
  identity by packed signature (a word reused as a node emits its identity
  once). An IDENTITY entry whose sig multi-token-splits registers its packed
  sig and marks it emitted (the main entry is the source identity).

Downstream consequence: the dialogue-script format (which mirrored the old
  per-component emissions turn-by-turn) was updated — `MHALL_TURNS` no longer
  encodes subword CANONIZES turns or component UNKNOWN asks; compound words
  are ratified as IDENTITY turns instead.

Spec: §8, §8.3, §11.3, §14.6, §14.11, §14.12; matrix KS-19/19a/38/39/43.
Tests: updated across test_ks*, test_ks_ast_emitter, test_ks_token_encoder,
  test_ks_compiler, and the dialogue fixtures (tests/_fixtures/__init__.py).

## Status

- [x] Spec (`specs/kscript.md`) rewritten.
- [x] Context (`CONTEXT.md`) — Target Significance term + binding-aware UNKNOWN (prior commit).
- [x] Task 1 — significance map.
- [x] Task 2 — ASTEmitter.
- [x] Task 3 — TokenEncoder.
- [x] Task 4 — Tests.
