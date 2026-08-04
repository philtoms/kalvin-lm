# Remove COMPOUND_TOKEN — Implementation Plan

**Spec:** @specs/kscript.md §11.3, @specs/kline.md §Structural Predicates, @specs/nlp_tokenizer.md, @specs/dialogue-driven-training.md §Decode, @specs/dialogue-cogitation.md
**Date:** 2026-08-04
**Status:** Done

## Goal

Eliminate `COMPOUND_TOKEN` entirely. A compound-word (a resolved word the
external BPE tokenizer splits into multiple subwords) becomes a single-token
self-referential identity `{S: [S]}` whose signature is the OR-reduction of
its subword tokens. The subwords live in the signature; no marker token
distinguishes a compound-word, and `COMPOUND_TOKEN` does not exist.

Signature sharing between a compound-word identity and a canon is accepted by
design — klines are classified by their own signature+nodes, and the
rationaliser ranges over all klines under a signature. No disambiguation.

See @specs/kscript.md §11.3 (rewritten) for the owning contract.

## Tasks

### Task 1 — Delete the constant (`src/kalvin/nlp_tokenizer.py`)
Remove `COMPOUND_TOKEN` and `COMPOUND_TOKEN_TYPE_WORD` and their docstrings.
Spec: @specs/nlp_tokenizer.md (Compound Marker Token section removed).

### Task 2 — Reduce structural predicates (`src/kalvin/kline.py`)
- Delete `is_compound_word` and the `from kalvin.nlp_tokenizer import COMPOUND_TOKEN`.
- `is_terminal`: empty or self-ref only.
- `is_identity`: self-ref only.
Spec: @specs/kline.md KL-26/26a/27 recast to self-ref.

### Task 3 — Compiler emits self-ref identity (`src/ks/token_encoder.py`)
`_emit_mts_for_tokens`: `packed = signature_of(tokens)` (no marker); the
compound-word identity kline is `{packed: [packed]}` (self-ref). Component
subword UNKNOWN klines still emitted. Drop the `COMPOUND_TOKEN` import and
the `compound_nodes` construction.
Spec: @specs/kscript.md §11.3.

### Task 4 — Dialogue rationaliser (`src/dialogue/rationalise.py`)
- `_cover_with_groundeds`: drop the `COMPOUND_TOKEN` stripping (marker never
  in nodes).
- `_reply_identity_ask`: the compound branch collapses into the general
  self-ref identity path. Drop `is_compound_word` / `COMPOUND_TOKEN` imports.
Spec: @specs/dialogue-cogitation.md.

### Task 5 — Dialogue decoder (`src/dialogue/decoder.py`)
Delete `compound_by_label`, `_maybe_catch_up_compound`, and the
`COMPOUND_TOKEN` import. A compound-word label resolves to its self-ref
identity directly from the label index; a block-canon under the same label
decodes verbatim.
Spec: @specs/dialogue-driven-training.md §Decode.

### Task 6 — Signifier docstring (`src/kalvin/signifier.py`)
Drop the `COMPOUND_TOKEN` reference in `signature_of`'s docstring.
Spec: @specs/signifier.md.

### Task 7 — Synthesizing trainer (`src/dialogue/synthesize.py`)
The trainer's compound-identity lookup (`_first_compound` / `is_compound_word`)
collapses into the self-ref identity path. Update accordingly.

### Task 8 — Tests
Update `tests/test_kline.py`, `tests/test_ks_token_encoder.py`,
`tests/test_ks_compiler.py`, `tests/test_expand.py`, and any dialogue
rationalise/decode tests. Compound-word identities are now self-ref
`{sig: [sig]}`; signatures are `signature_of(subwords)` with no marker.

## Test Mapping

| Spec ID | Test file | Status |
| ------- | --------- | ------ |
| KL-26/26a/27 (recast) | tests/test_kline.py | pending |
| §11.3 (self-ref compound) | tests/test_ks_token_encoder.py, tests/test_ks_compiler.py | pending |
| §11.3 (S1 compound) | tests/test_expand.py | pending |
| dialogue decode (no catch-up) | tests/test_dialogue_decode* | pending |
| dialogue rationalise | tests/test_rationalise* | pending |

## Design Decisions

- **No signature disambiguation.** A compound-word identity and a canon may
  share a signature. Accepted; the rationaliser ranges over the bucket.
  Rationale: follows the existing `dict[sig, list[KLine]]` model; klines are
  classified independently.
- **Block-canon no longer fuses with the compound identity.** `had => did
  have` decodes to its declared relationship kline verbatim. The old
  `_maybe_catch_up_compound` existed only to re-prepend the marker; with no
  marker, there is nothing to catch up to.
