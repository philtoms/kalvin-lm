# NLP Tokenizer Extraction Implementation Plan

**Parent:** finishing the tokenizer/signifier isolation — moving all
bit-pattern and type methods off the base `Tokenizer` and the `KTokenizer`
interface into `NLPTokenizer`, so that base + interface are layout- and
type-agnostic (parallel to the signifier extraction).
**Status:** spec complete; implementation pending
**Spec refs:** `@specs/tokenizer.md` (interface, layout-agnostic),
`@specs/nlp_tokenizer.md` (production concrete, owns layout/types/engine)
**Depends on:** the signifier extraction (already merged)

## Problem

The base `Tokenizer` (`src/kalvin/tokenizer.py`) currently implements the
NLP node layout — `encode` (`<<32 |` packing), `decode`
(`& 0xFFFFFFFF` unpacking), the type dictionary, `lookup_type` /
`lookup_type_entry`, `UNKNOWN_TYPE`, `type_size`, `batch_encode` — and
declares `class Tokenizer(KTokenizer)`. The `KTokenizer` interface
additionally defined `lookup_type` / `lookup_type_entry` as default
methods. Two consequences:

1. The base `Tokenizer` (a BPE engine wrapper) is presented as a
   `KTokenizer`, but it carries NLP-specific layout. A non-NLP tokenizer
   could not share the base without inheriting the NLP packing.
2. The ks compiler (`src/ks/token_encoder.py:281`) reaches into the node
   layout directly (`sig_uint64 & 0xFFFFFFFF`) to call `lookup_type_entry`,
   because that method takes a BPE id, not a node — leaking the layout into
   the compiler.

The spec layer (already updated) now states: `KTokenizer` is layout- and
type-agnostic; the base `Tokenizer` is a BPE-engine wrapper, **not** a
`KTokenizer`; `NLPTokenizer` is the sole concrete `KTokenizer` and owns
the layout, types, and engine foundation. This plan brings the code to
that state.

## Spec References

- `@specs/tokenizer.md` — `KTokenizer` interface (role: text↔nodes); BPE
  engine as foundation (not a KTokenizer).
- `@specs/nlp_tokenizer.md` — production concrete; owns encode/decode,
  `(nlp_type32 << 32) | bpe_token_id` layout, type dictionary,
  `lookup_type*`, BPE-engine inheritance.
- `@specs/kscript.md` §11.5 — node opacity at the compiler; the
  type-info leak is the violation this plan fixes.

## Module Layout (target)

```
kalvin/
  abstract.py       — KTokenizer (encode/decode/vocab_size; NO lookup_type*)
  tokenizer.py      — Tokenizer: BPE-engine wrapper (NOT a KTokenizer)
                      train/save/load, encode_bpe, decode_bpe, vocab_size
  nlp_tokenizer.py  — NLPTokenizer(Tokenizer): sole concrete KTokenizer
                      encode/decode (packing), type dict, lookup_type*,
                      batch_encode, type_size, UNKNOWN_TYPE
```

## Implementation Tasks

### Task 1 — Shrink base `Tokenizer` to a BPE-engine wrapper (`src/kalvin/tokenizer.py`)

- **Spec ref:** @tokenizer §BPE Engine; @nlp_tokenizer §BPE Engine
  Foundation.
- Drop `class Tokenizer(KTokenizer)` → `class Tokenizer:` (plain class; a
  BPE engine wrapper, not a `KTokenizer`). Remove the `KTokenizer` import.
- **Move to NLPTokenizer** (Task 2): `encode`, `decode`, `lookup_type`,
  `lookup_type_entry`, `batch_encode`, `type_size`, `UNKNOWN_TYPE`, the
  `_types` instance attribute, and the `types` constructor param.
- **Keep** (engine surface): `__init__(self, bpe=None)` (drop the `types`
  param), `_check_available`, `train`, `save_to_directory`,
  `_load_bpe_engine`, `encode_bpe`, `vocab_size`.
- **Add `decode_bpe(self, bpe_ids: list[int]) -> str`** — raw BPE IDs →
  text, symmetric to `encode_bpe`. This is the engine-level decode; the
  layout-unpacking `decode` moves to NLPTokenizer and calls `decode_bpe`
  after stripping the high 32 bits.
- **Remove `from_directory`** (the public classmethod) — zero callers
  (verified). `_load_bpe_engine` stays (used by NLPTokenizer). The
  docstring's mention of `from_directory` is updated to `load_from_directory`
  semantics or removed.
- Module docstring: rewrite to describe the BPE-engine wrapper role, not
  the typed-node layout (which moves to @nlp_tokenizer).

### Task 2 — `NLPTokenizer` gains the moved methods (`src/kalvin/nlp_tokenizer.py`)

- **Spec ref:** @nlp_tokenizer §Interface, §Type Dictionary.
- Move `encode`, `decode`, `batch_encode`, `lookup_type`,
  `lookup_type_entry`, `type_size`, `UNKNOWN_TYPE` from base into
  `NLPTokenizer`. The `_types` dict and the `types` constructor param move
  here too.
- `encode` / `decode` / `lookup_type*` keep their bodies verbatim (the
  `(sig_word << 32) | bpe_id` packing and the `& 0xFFFFFFFF` unpacking now
  live here, their rightful home).
- `__init__` signature becomes `__init__(self, bpe=None, types=None)` for
  the base-composed form; the existing production constructor
  `NLPTokenizer(tokenizer_path=None, tokenizer_name=...)` already builds
  `bpe` + `types` and calls `super().__init__(...)` — update the super call
  to the new base signature (which no longer takes `types`), then set
  `self._types` directly on NLPTokenizer. (Decision: whether NLPTokenizer
  stores `_types` itself or passes to base — since base no longer has
  `_types`, NLPTokenizer owns it.)
- `decode` calls `self.decode_bpe(...)` (the new base method) after
  unpacking, instead of reaching into `self._bpe.decode` directly.

### Task 3 — `KTokenizer` interface (`src/kalvin/abstract.py`)

- **Spec ref:** @tokenizer §Interface.
- Already trimmed in the spec pass: `lookup_type` / `lookup_type_entry`
  removed; docstring updated. **Verify** `__all__` and that `KTokenizer`
  remains abstract with `encode`/`decode`/`vocab_size` only.
- Confirm `test_abstract.py` still passes (it checks KTokenizer is abstract
  + in `__all__`).

### Task 4 — Fix Site B: the compiler node-layout leak (`src/ks/token_encoder.py`)

- **Spec ref:** @kscript §11.5 (node opacity); @nlp_tokenizer §Type
  Dictionary Lookup.
- Add a **node-taking accessor** to `NLPTokenizer`:
  `lookup_type_entry_for_node(node: int) -> dict | None` — unpacks the node
  internally (its own `& 0xFFFFFFFF`) and delegates to `lookup_type_entry`.
  The layout knowledge stays inside NLPTokenizer.
- In `_make_kdbg` (line ~281), replace:
  ```python
  entry = self._tokenizer.lookup_type_entry(sig_uint64 & 0xFFFFFFFF)
  ```
  with a **gated** call that acknowledges type-info as an NLP-specific
  debug affordance:
  ```python
  entry = None
  lookup = getattr(self._tokenizer, "lookup_type_entry_for_node", None)
  if lookup is not None:
      entry = lookup(sig_uint64)
  ```
  No `& 0xFFFFFFFF` in the compiler; the KTokenizer interface stays clean
  (the method isn't on it); the path is honestly gated.
- The `decode` call on the line above is fine (it's on the interface).

### Task 5 — Tests

- **Spec ref:** @nlp_tokenizer §Test Matrix (TOK-8/10/11/12, TOK-NLP-1..14).
- `tests/test_ks_token_encoder.py` — `MockMultiTokenTokenizer` already
  gained `lookup_type_entry` in the spec pass; with Task 4's gate it
  should gain `lookup_type_entry_for_node` returning `None` instead (or
  rely on the `getattr` default). Align with the chosen gate form.
- `tests/test_nlp_tokenizer.py` — verify encode/decode/lookup_type tests
  still pass against NLPTokenizer directly (they should, unchanged — the
  methods moved but the call surface is identical).
- `tests/test_default_tokenizer.py` — references `Tokenizer.from_directory`
  in a docstring; update the docstring (from_directory removed). Verify no
  code path calls it.
- `tests/test_abstract.py` — verify KTokenizer ABC test passes.
- **No new behavioural tests** — this is behaviour-preserving relocation
  plus the Site B gate (which is structurally equivalent for type-aware
  tokenizers and yields empty type_info for non-type-aware ones).

## Test Mapping

| Spec ID | Test (file) | Status |
| ------- | ----------- | ------ |
| TOK-7   | `test_nlp_tokenizer.py` encode/decode empty | unchanged |
| TOK-8   | `test_nlp_tokenizer.py` typed-node format | unchanged (method moved) |
| TOK-9   | `test_nlp_tokenizer.py` round-trip | unchanged |
| TOK-10  | `test_nlp_tokenizer.py` lookup_type | unchanged (method moved) |
| TOK-11  | `test_nlp_tokenizer.py` unknown fallback | unchanged |
| TOK-12  | `test_abstract.py` + structural (base not KTokenizer) | verify/extend |
| TOK-NLP-14 | structural: NLPTokenizer sole concrete KTokenizer | new/verify |

## Design Decisions

| Decision | Outcome | Rationale |
| --- | --- | --- |
| Fork | (i) base drops `KTokenizer`; NLPTokenizer sole concrete | base is a BPE wrapper, not a text↔node tokenizer; nothing constructs bare Tokenizer as a KTokenizer |
| Where the layout lives | NLPTokenizer owns encode/decode/`_types`/`UNKNOWN_TYPE` | bit-packing is the NLP operation, parallel to NLPSignifier owning the bit-algebra |
| `decode_bpe` on base | add it (symmetric to `encode_bpe`) | base owns raw BPE↔text; NLPTokenizer's `decode` unpacks then delegates — clean layering |
| `from_directory` | remove (zero callers) | dead; `_load_bpe_engine` is the live engine loader |
| Site B fix | node-taking accessor + getattr gate | removes `& 0xFFFFFFFF` from compiler; keeps KTokenizer interface clean; honestly marks type-info as NLP-debug affordance |
| Behaviour | preserving relocation + one gated path | no semantic change for production (NLPTokenizer); non-type-aware tokenizers get empty type_info |

## Deferred

None new. The remaining signifier-extraction deferrals (signature-zero
sentinel; bare-`&` overlap/gap class) are unchanged and tracked in
`specs/signifier.md §Opacity Invariant`.

## Status

- [x] Spec (`specs/tokenizer.md`, `specs/nlp_tokenizer.md`, `abstract.py`
      docstring/method trim — done in the spec pass)
- [x] Plan (this document)
- [x] Task 1: shrink base `Tokenizer`
- [x] Task 2: move methods to `NLPTokenizer`
- [x] Task 3: verify `KTokenizer` interface
- [x] Task 4: fix Site B (compiler leak)
- [x] Task 5: tests
