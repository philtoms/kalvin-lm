# Signifier Implementation Plan

**Parent:** isolating signature bit-operations behind an injectable
Signifier, so that Kalvin core is agnostic to the NLP bit-algebra (and to bit
algebra at all)
**Status:** spec complete; implementation pending
**Spec refs:** `@specs/signifier.md` (new — `KSignifier` interface +
NLPSignifier), `@specs/signature.md` (trimmed — delegates algebra to
@signifier)
**Depends on:** —

## Problem

`kalvin.signature` implements the NLP bit-algebra — `make_signature`
(OR-reduce) and `signifies` (masked `TYPE_MASK` AND over the upper 32 bits) —
as free functions called directly from ~15 sites across Kalvin core. Kalvin is
therefore aware of, and coupled to, one concrete signature mechanism. The
recent tokenizer refactor isolated node *construction* behind `KTokenizer`;
this plan finishes the job for signature *reduction and matching* by
introducing a parallel `KSignifier` interface with `NLPSignifier` as its sole
concrete implementation.

The interface specifies role and types only (see @signifier §Interface); all
algebraic properties live in NLPSignifier. Kalvin core treats node and
signature values as opaque.

## Spec References

- `@specs/signifier.md` — `KSignifier` interface (`make_signature`,
  `signifies`), NLPSignifier concrete, Opacity Invariant, SIG-1..SIG-16
- `@specs/signature.md` — signature as a value; delegates creation/matching to
  @signifier
- `@specs/tokenizer.md`, `@specs/nlp_tokenizer.md` — NLP bundle = NLPTokenizer
  + NLPSignifier; peer relationship
- `@specs/agent.md`, `@specs/model.md`, `@specs/stm.md`, `@specs/cogitator.md`
  — consume `make_signature` / `signifies` via the injected Signifier

## Module Layout (target)

```
kalvin/
  abstract.py       — KTokenizer, KSignifier (new ABC)
  signifier.py      — NLPSignifier (sole concrete; the masked algebra)  [NEW]
  signature.py      — DELETED (replaced by signifier.py)
  kline.py          — KLine data + pure-structure predicates (is_identity,
                      is_canon); is_canon takes a signifier
  tokenizer.py      — Tokenizer (unchanged operationally)
  nlp_tokenizer.py  — NLPTokenizer (unchanged operationally)
```

No new `nlp_signifier.py` — `NLPSignifier` lives in `signifier.py` alongside
no base concrete (the interface is the ABC; there is no generic Signifier).

## Implementation Tasks

### Task 1 — `KSignifier` ABC (`src/kalvin/abstract.py`)

- **Spec ref:** @specs/signifier.md §Interface (`KSignifier`)
- Add `KSignifier(ABC)` alongside the existing `KTokenizer`. The abstract
  methods declare **role and types only** — no algebraic guarantees:
  ```python
  class KSignifier(ABC):
      @abstractmethod
      def make_signature(self, nodes: Sequence[int]) -> int: ...
      @abstractmethod
      def signifies(self, a: int, b: int) -> bool: ...
  ```
- Add `KSignifier` to `__all__`. Do **not** add `lookup_type`-style extras;
  the interface is exactly the two operations.

### Task 2 — `NLPSignifier` (`src/kalvin/signifier.py`, new)

- **Spec ref:** @specs/signifier.md §NLPSignifier, §Properties
- Move the contents of `signature.py` into a class. The two constants and two
  methods relocate verbatim (no behaviour change):
  ```python
  class NLPSignifier(KSignifier):
      _TYPE_MASK = 0xFFFF_FFFF_0000_0000

      def make_signature(self, nodes):           # sig |= node  (OR-reduce)
      def signifies(self, a, b):                 # (a & b & _TYPE_MASK) != 0
  ```
- Module docstring documents the NLP mechanism (nlp_type32 packing, masked
  type-word overlap) and cross-refs @nlp_tokenizer. Keep the rationale comment
  that currently lives in `signature.py` about why the BPE half is masked.
- This is the sole concrete Signifier; it is the production implementation and
  the NLP bundle's signifier member.

### Task 3 — Delete `src/kalvin/signature.py`

- Remove the module entirely. Its contents now live in `signifier.py` (Task 2)
  behind the `KSignifier` interface (Task 1).

### Task 4 — Signifier injection through the wiring chain

- **Spec ref:** @specs/signifier.md (injectable concern; parallel to
  @tokenizer)
- Thread `signifier: KSignifier | None = None` through every constructor that
  already takes `tokenizer`, defaulting to `NLPSignifier()`:
  - `src/kalvin/agent.py` `KAgent.__init__` — add `signifier` param next to
    `tokenizer`; add `_default_signifier()` mirroring `_default_tokenizer()`;
    store `self._signifier`; expose a `signifier` property.
  - `src/training/trainer/trainer.py` `Trainer.__init__` — add `signifier`
    param next to `tokenizer`; store `self._signifier`.
  - `src/training/harness/__main__.py` — construct `NLPSignifier()` next to
    `shared_tokenizer`; pass to `KAgent(...)` and `Trainer(...)`.
  - `src/ks/compiler.py`, `src/ks/__init__.py`, `src/ks/token_encoder.py`,
    `src/training/harness/adapter.py` — add `signifier` param next to
    `tokenizer` where they construct or hold a tokenizer for compilation.
- **Decision rule for who needs a signifier:** any component that today calls
  `make_signature` or `signifies` (see Task 5). Components that only encode/
  decode text (pure tokenizer consumers) do not need one.

### Task 5 — Migrate call sites (free functions → injected signifier)

- **Spec ref:** @specs/signifier.md §Interface
- Replace every `make_signature(...)` / `signifies(...)` call with
  `self._signifier.make_signature(...)` / `self._signifier.signifies(...)`.
  **Behaviour-preserving relocation only** — the deferred bare-bitwise bugs
  (Task 7) are NOT touched in this migration. Sites:

  | File | Current | Replacement |
  | ---- | ------- | ----------- |
  | `src/kalvin/agent.py` | `make_signature` (3×: prepare, expected_sig, reciprocal) | `self._signifier.make_signature` |
  | `src/kalvin/model.py` | `make_signature` (×1, line 177), `signifies` (×1, line 399) | via `self._signifier` (the Model must hold a signifier — add in Task 4) |
  | `src/kalvin/expand.py` | `make_signature` (×3: lines 177, 221, 433), `signifies` (×2: lines 286, 314) | via a `signifier` param on the expand functions (see Decision: threading) |
  | `src/kalvin/misfit.py` | `make_signature` (×4) | via a signifier param on `classify_misfit` / `generate_expansions` |
  | `src/kalvin/stm.py` | `make_signature` (×3) | `self._signifier.make_signature` (STM holds a signifier — add in Task 4) |
  | `src/kalvin/kline.py` | `make_signature` in `is_canon`, `_infer_level`, `_infer_op_symbol` | `signifier.make_signature` (see Task 6) |
  | `src/ks/token_encoder.py` | `make_signature` (×2) | `self._signifier.make_signature` |
  | `src/training/trainer/trainer.py` | `make_signature` (line ~282, gap computation) | `self._signifier.make_signature` |

- **Decision: threading `expand`/`misfit` (module-level functions).** These are
  not classes, so they cannot store `self._signifier`. Two options:
  - **(a)** pass `signifier` as an explicit parameter to every `expand`/misfit
    function (mirrors how they already take `model`);
  - **(b)** refactor `expand`/`misfit` into classes holding a signifier.
  - **Decision (confirmed): option (a)** — smaller blast radius, matches the
    existing `model`-parameter convention.

### Task 6 — `is_canon` and the inference helpers

- **Spec ref:** @specs/signature.md (structural predicate), @specs/signifier.md
  §What a Signifier is Not (is_canon delegates reduction)
- `is_identity(kline)` — **unchanged**. Pure structure (empty nodes or
  self-referential); no bit manipulation. Stays a free function taking no
  signifier.
- `is_canon(kline, signifier)` — add a `signifier` parameter; body becomes
  `kline.signature == signifier.make_signature(kline.nodes)`. Update every
  `is_canon(...)` call site to pass the signifier.
- `_infer_level(kline, signifier)` and `_infer_op_symbol(kline, signifier)` —
  add `signifier` param; route `make_signature` through it. (These also contain
  deferred bare-`&` overlap — Task 7 — which stays untouched here except for
  the `make_signature` relocation.)
- **Threading note:** `is_canon` / `_infer_level` are called from `kline_display`
  and `_infer_op_symbol`. `kline_display(kline, tokenizer)` must become
  `kline_display(kline, tokenizer, signifier)`. Audit its call sites.

### Task 7 — Tests

- **Spec ref:** @specs/signifier.md §Test Matrix (SIG-1..SIG-16)
- **Rename** `tests/test_signature.py` → `tests/test_signifier.py`. Update
  imports to `from kalvin.signifier import NLPSignifier`. Construct one
  `NLPSignifier()` instance per test class (or a fixture) and call
  `signifier.make_signature` / `signifier.signifies`. The existing assertions
  are all NLPSignifier properties and stay valid unchanged. Add the `T()`
  helper's doc note that it is an NLPSignifier-specific test aid.
- **`tests/test_abstract.py`** — add an instantiation/ABC test for `KSignifier`
  (abstract; cannot instantiate; `NLPSignifier` satisfies it), mirroring
  whatever exists for `KTokenizer`.
- **Migrate test imports:** the following test files import
  `make_signature` and must switch to an `NLPSignifier()` instance:
  `test_agent.py`, `test_cogitator_drain.py`, `test_cogitator_handler.py`,
  `test_countersign_resolution.py`, `test_encode_text.py`,
  `test_ks_token_encoder.py`, `test_model.py`, `test_nlp_tokenizer.py`.
  For test ergonomics, consider a shared `signifier` fixture in
  `tests/conftest.py` (parallel to any tokenizer fixture).
- **No new behavioural tests** — this plan is behaviour-preserving relocation.
  Existing test suites must pass unchanged (modulo import/source rewiring).

## Test Mapping

| Spec ID | Test (file) | Status |
| ------- | ----------- | ------ |
| SIG-1  | `test_signifier.py` `test_empty_nodes` / `test_well_known_zero` | migrated from `test_signature.py` |
| SIG-4  | `test_signifier.py` `test_single_node` / `test_identity` | migrated |
| SIG-6  | `test_signifier.py` `test_multiple_nodes` / `test_or_reduce` | migrated |
| SIG-7  | `test_signifier.py` `test_zero_signifies_nothing` | migrated |
| SIG-9  | `test_signifier.py` `test_overlapping_type_bits` | migrated |
| SIG-10 | `test_signifier.py` `test_non_overlapping_type_bits` | migrated |
| SIG-14 | `test_signifier.py` `test_or_reduction_of_packed_nodes` | migrated |
| SIG-15 | `test_signifier.py` `test_bpe_component_masked_off` | migrated |
| SIG-16 | `test_signifier.py` `test_type_overlap_beats_bpe_difference` | migrated |
| (ABC)  | `test_abstract.py` KSignifier instantiation | new |

## Design Decisions

| Decision | Outcome | Rationale |
| --- | --- | --- |
| Interface vs concrete split | `KSignifier` ABC = role/types only; `NLPSignifier` sole concrete | @signifier §Interface — a future probabilistic/learned Signifier must violate nothing in the interface |
| No base concrete Signifier | only `NLPSignifier` exists | the bit-algebra *is* the NLP operation; there is no generic mechanism to put in a base class |
| Module name | `signifier.py` (not `nlp_signifier.py`) | no sibling concrete; the concrete IS NLP but the file holds the only impl |
| Injection channel | alongside existing `tokenizer` param | same wiring convention as the tokenizer refactor; no new top-level parameter concept |
| `expand`/`misfit` threading | explicit `signifier` param (option a) | matches existing `model`-param convention; avoids a refactor into classes |
| `is_canon` location | stays in `kline.py`, takes a signifier | structural predicate; the only bit-op is the delegated `make_signature` |
| `is_identity` location | stays in `kline.py`, no signifier | pure structure, no bit manipulation |
| Behaviour of migrated calls | unchanged (relocation only) | deferred bare-bitwise bugs fixed separately (Task 7 deferrals) |
| Test file rename | `test_signature.py` → `test_signifier.py` | the module under test is renamed; all criteria are NLPSignifier properties |
| `NLPSignifier()` default | constructed in `_default_signifier()` and `__main__` | parallels `_default_tokenizer()` / `NLPTokenizer()` |

## Deferred (Opacity Invariant violations — see @signifier §Opacity Invariant)

These are tracked in the spec and are **not** part of this plan. Each is a
separate follow-up:

1. **Signature-zero sentinel** (`agent.py`, `expand.py`, `stm.py`):
   `signature == 0` means "empty/unset." This hardcodes NLPSignifier's
   empty-set value into Kalvin core. Remedy: a structural/unset mechanism, not
   a routed Signifier call. *(Highest-impact opacity leak — would break a
   learned signifier first.)*
2. **Bare-bitwise overlap/gap class** — should route through `signifies`:
   - `stm.py:126` STM candidate query (bare `&`; actually wants a Tokenizer
     bottom-word compare — flagged Tokenizer cleanup);
   - `misfit.py` contributor lookup (`& gap`, lines ~65, 104) and
     `classify_misfit` gap (`& ~`, lines 30–31);
   - `expand.py:434-435` gap (`signature & ~nodes_sig`);
   - `trainer.py:283-284` gap (`signature & ~nodes_sig`);
   - `kline.py:258,260` `_infer_level` bare `&`.
3. **Tokenizer node-unpack leak** (`token_encoder.py:280`, `& 0xFFFFFFFF`):
   the compiler reaches into the node layout. Remedy: a Tokenizer accessor for
   "unpack a node." Flagged Tokenizer cleanup, parallel to this effort.

## Status

- [x] Spec (`specs/signifier.md` new; `specs/signature.md` trimmed; cross-refs
      in tokenizer/nlp_tokenizer/model/agent/kline/stm/cogitator/kscript/harness)
- [x] Plan (this document)
- [x] Task 1: `KSignifier` ABC
- [x] Task 2: `NLPSignifier`
- [x] Task 3: delete `signature.py`
- [x] Task 4: injection chain
- [x] Task 5: migrate call sites
- [x] Task 6: `is_canon` + helpers
- [x] Task 7: tests
