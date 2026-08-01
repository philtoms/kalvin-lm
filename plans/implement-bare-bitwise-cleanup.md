# Bare-`&` Opacity Cleanup Implementation Plan

**Parent:** the last deferred opacity violation — routing every bare-bitwise
operation on signature/node values in Kalvin core behind the Signifier.
Closes the final item from `specs/signifier.md §Opacity Invariant`.
**Status:** spec complete; implementation pending
**Spec refs:** `@specs/signifier.md` (gains `residual` + `classify_misfit`;
§Opacity Invariant loses its last deferred bullet), `@specs/model.md`
(MOD-52..55 repoint), `@specs/cogitator.md` (misfit sections delegate)
**Depends on:** the signifier extraction, signature-zero removal (merged)

## Problem

Bare bitwise operations on signature/node values remain in five Kalvin
modules. They split into two operations:

- **Residual (set-difference)** — `S & ~N`: the type-word bits a signature
  claims that its nodes don't cover. Sites:
  `expand.py:437-438`, `trainer.py:292-293`.
- **Overlap** — `k.signature & gap` / `sig & nodes`: type-word overlap.
  Sites: `misfit.py:66,80,105`, `stm.py:131`, `kline.py:258,260`.

Both are NLP bit-algebra and must live behind the Signifier. Additionally,
`classify_misfit` currently lives in `misfit.py` but is a Signifier concern;
it moves onto `KSignifier`.

Per the spec pass (already merged into the working tree): `KSignifier` gains
`residual(a, b)` and `classify_misfit(signature, nodes)`; `NLPSignifier`
implements both with type-word masking (consistent with `signifies`).
`misfit.py`'s `classify_misfit` is deleted; `generate_expansions` + helpers
remain, now fed residual values and calling `signifies` for overlap.

## Behavior change (endorsed)

Switching the overlap sites to `signifies` (masked) and `classify_misfit` to
masked residual fixes latent BPE-collision false-positives: same-type,
different-token pairs (e.g. a connoted `{Noun_A: [Noun_B]}`) no longer
classify as misfit. This is a **real behavior change** — it will surface in
tests that constructed such cases expecting misfit classification.

## Spec References

- `@specs/signifier.md` §Interface (`residual`, `classify_misfit`),
  §NLPSignifier (masked implementations), §Test Matrix (SIG-17..23),
  §Opacity Invariant (no deferred bullets remain).
- `@specs/model.md` MOD-52..55 — repoint to @signifier.
- `@specs/cogitator.md` §Misfit Classification — delegates to @signifier.

## Implementation Tasks

### Task 1 — `KSignifier` ABC + `NLPSignifier` (`src/kalvin/abstract.py`, `src/kalvin/signifier.py`)

- **Spec ref:** @signifier §Interface, §NLPSignifier.
- Add two abstract methods to `KSignifier`:
  ```python
  @abstractmethod
  def residual(self, a: int, b: int) -> int: ...
  @abstractmethod
  def classify_misfit(self, signature: int, nodes: Sequence[int]) -> tuple[bool, bool]: ...
  ```
- Implement both in `NLPSignifier`:
  ```python
  def residual(self, a, b):
      return (a & ~b) & _TYPE_MASK
  def classify_misfit(self, signature, nodes):
      nodes_sig = self.signature_of(nodes)
      underfit = self.residual(signature, nodes_sig) != 0
      overfit = self.residual(nodes_sig, signature) != 0
      return underfit, overfit
  ```
- Update the `KSignifier` docstring (signatures → abstract methods list).

### Task 2 — Delete `misfit.classify_misfit`; update callers

- **Spec ref:** @signifier §classify_misfit.
- Delete `classify_misfit` from `src/kalvin/misfit.py`.
- Update the two callers to use `signifier.classify_misfit(signature, nodes)`
  (value-space, per the agreed signature):
  - `src/kalvin/expand.py:429` — `classify_misfit(candidate, signifier)` →
    `signifier.classify_misfit(candidate.signature, candidate.nodes)`.
  - `src/training/trainer/trainer.py:290` —
    `classify_misfit(target, self._signifier)` →
    `self._signifier.classify_misfit(target.signature, target.nodes)`.
- Drop the now-unused `classify_misfit` import from both files.

### Task 3 — Route residual producers through `signifier.residual`

- **Spec ref:** @signifier §residual.
- `src/kalvin/expand.py:437-438`:
  ```python
  underfit_gap = signifier.residual(candidate_sig, nodes_sig)
  overfit_mask = signifier.residual(nodes_sig, candidate_sig)
  ```
- `src/training/trainer/trainer.py:291-293`:
  ```python
  target_nodes_sig = self._signifier.signature_of(target.nodes)
  underfit_gap = self._signifier.residual(target.signature, target_nodes_sig)
  overfit_mask = self._signifier.residual(target_nodes_sig, target.signature)
  ```
  (These feed the LLM decision-request `misfit` dict; the values are now
  masked residuals — consistent with `classify_misfit`.)

### Task 4 — Route overlap sites through `signifier.signifies`

- **Spec ref:** @signifier §signifies.
- `src/kalvin/misfit.py:66,105` —
  `model.where(lambda k: (k.signature & gap) != 0)` →
  `model.where(lambda k: signifier.signifies(k.signature, gap))`.
  (`signifier` is already threaded through `generate_expansions` and its
  helpers from the earlier signifier extraction.)
- `src/kalvin/misfit.py:80` —
  `[n for n in kline.nodes if (n & excess) != 0]` →
  `[n for n in kline.nodes if signifier.signifies(n, excess)]`.
  (`_split_excess` gains a `signifier` param.)
- `src/kalvin/stm.py:131` —
  `(kline.signature & sig) != 0` → `self._signifier.signifies(kline.signature, sig)`.
  (STM holds a signifier from the earlier extraction.)
- `src/kalvin/kline.py:258,260` —
  `_infer_level`'s `signature & nodes[0]` / `signature & nodes_sig` →
  `signifier.signifies(...)`. (`_infer_level` already takes `signifier`.)
- Update `model.py:401` docstring (the `& sig != 0` mention) to reference
  `signifies`.

### Task 5 — Tests

- **Spec ref:** @signifier SIG-17..23.
- `tests/test_signifier.py` — add test classes for `residual` (SIG-17..19)
  and `classify_misfit` (SIG-20..23). Use the existing `T()` helper.
- `tests/test_misfit.py` — its `classify_misfit(k, signifier)` calls move
  to `signifier.classify_misfit(k.signature, k.nodes)`. **Review each test
  case** for the behavior change: any case that constructed a same-type,
  different-token pair expecting misfit will now classify as canonical
  (False, False). Update the assertion or the test data accordingly. The
  `generate_expansions` tests may also change (different contributors found
  once overlap is masked).
- `tests/test_agent.py:572` — `classify_misfit(k, signifier)` →
  `signifier.classify_misfit(k.signature, k.nodes)`.
- `tests/test_expand.py` — verify expansion tests still pass; if any
  constructed BPE-collision false-positives, update the expected behavior.
- **No new behavioral tests beyond SIG-17..23** — the rest is relocation
  + the endorsed behavior fix.

## Test Mapping

| Spec ID | Test (file) | Status |
| ------- | ----------- | ------ |
| SIG-17..19 | `test_signifier.py` `TestResidual` | new |
| SIG-20..23 | `test_signifier.py` `TestClassifyMisfit` | new |
| MOD-52..55 | `test_misfit.py` (via @signifier) | review for behavior change |

## Design Decisions

| Decision | Outcome | Rationale |
| --- | --- | --- |
| `residual` on interface | yes, `(a, b) → int` | valid signature-algebra op parallel to signature_of/signifies |
| `classify_misfit` on interface | yes, `(signature, nodes) → (bool, bool)` | encapsulates residual + `!= 0`; callers never see residual representation |
| Masking | masked (consistent with `signifies`) | type-word-only; fixes latent BPE-collision bugs |
| `classify_misfit` signature | value-space `(signature, nodes)` | consistent with other interface methods (all take raw uint64) |
| `misfit.classify_misfit` | delete | moved to Signifier |
| `generate_expansions` location | stays in `misfit.py` | rationalisation pipeline, not algebra |

## Deferred

None. This job closes the last item in `specs/signifier.md §Opacity Invariant`
— after it, there are no known opacity violations.

## Status

- [x] Spec (`signifier.md`, `model.md`, `cogitator.md`)
- [x] Plan (this document)
- [x] Task 1: `KSignifier` + `NLPSignifier` methods
- [x] Task 2: delete `misfit.classify_misfit`; update callers
- [x] Task 3: route residual producers
- [x] Task 4: route overlap sites
- [x] Task 5: tests
