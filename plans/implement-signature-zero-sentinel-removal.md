# Signature-Zero Sentinel Removal Implementation Plan

**Parent:** removing Kalvin's reliance on the value `0` as a special
signature (the highest-impact opacity leak from the signifier extraction —
it would break a learned signifier whose empty-set signature is not `0`).
**Status:** spec complete; implementation pending
**Spec refs:** `@specs/agent.md` §Phase 1: Prepare (signatures must arrive
set), `@specs/signifier.md` §Opacity Invariant (sentinel removed from
deferred list)
**Depends on:** the signifier extraction (already merged)

## Problem

Four sites in Kalvin core treat the signature value `0` as special:

| Site | Meaning | Status |
| ---- | ------- | ------ |
| `agent.py:202` | `if signature == 0 and nodes: derive` | lazy-derive (transient) |
| `agent.py:206` | `if signature != 0 and grounded(...)` | guard for the above |
| `expand.py:178` | `if sig == 0: break` (after `make_signature`) | unreachable defensive |
| `stm.py:125` | `if sig == 0: return []` | harmless early-return |

This hardcodes NLPSignifier's empty-set value into Kalvin core. A Signifier
whose empty-set signature is not `0` would break all four.

Investigation (grounded): the "unset" meaning is **transient** — production
never relies on the lazy-derive. Every production caller pre-computes the
signature (`scripts/encode_text.py`, `scripts/kalvin_test.py`, the harness
adapter via the compiler, the agent's own countersign/reciprocal path).
Only a few tests hand-construct `KLine(0, [nodes])` as a placeholder, and
even those pre-compute immediately after.

The decision: **remove all four sentinel checks and let Kalvin crash on
misuse.** A single boundary `assert` at `rationalise` entry makes the crash
loud (a "must arrive set" contract), but it is not a value-test — it does
not special-case `0`; `0` becomes an ordinary signature value. Callers are
responsible for computing the signature before rationalising.

## Spec References

- `@specs/agent.md` §Phase 1: Prepare — the contract (signatures must
  arrive set; rationalise asserts; no derivation, no special value).
- `@specs/signifier.md` §Opacity Invariant — sentinel removed from the
  deferred list (now only the bare-`&` overlap/gap class remains).
- `@specs/kline.md` — signature is "required, uint64" (already consistent).

## Implementation Tasks

### Task 1 — Remove the four sentinel checks

- **`src/kalvin/agent.py`** — replace the conditional derive + `!= 0`
  guard (lines ~202–206) with a boundary assert and an unconditional
  ground check:
  ```python
  # Phase 1: Prepare — callers must provide a set signature.
  assert kline.signature is not None, (
      "KLine.signature must be set before rationalise; callers compute it "
      "via signifier.make_signature(nodes)."
  )
  # Phase 2: Ground check (Frame/LTM/Base only — not STM)
  if self._model.grounded(kline):
      self._model.add_to_stm(kline)
      self._publish("ground", kline, kline, D_MAX)
      return True
  ```
  The `is not None` assertion is a presence check, not a value-test — `0`
  is no longer special.
- **`src/kalvin/expand.py`** — remove the `if sig == 0: break` at line ~178
  (unreachable: the preceding `is_identity(kline)` already catches empty
  nodes; a non-identity kline reaching here has nodes).
- **`src/kalvin/stm.py`** — remove the `if sig == 0: return []` at line ~125
  (let the scan run; an empty-signature query simply matches nothing).

### Task 2 — Fix tests that rely on the lazy-derive

Identified sites hand-constructing `KLine(0, [nodes])` and relying on the
removed derive:

- `tests/test_cogitator_handler.py:94` — already pre-computes immediately
  after; the `KLine(0, [...])` placeholder can be replaced with the
  computed signature directly (or `KLine(signifier.make_signature([...]), [...])`).
- Scan `tests/test_countersign_resolution.py`, `tests/test_cogitator_drain.py`,
  `tests/test_agent.py` for any `KLine(0, [non-empty nodes])` passed to
  rationalise; replace with a pre-computed signature.
- Identity klines (`KLine(0, [])`) are fine — `0` is a legitimate signature
  for an empty node set; these do not rely on the derive and must **not**
  be changed. The assertion passes them (signature is set, to `0`).

Decision rule for tests: change a `KLine(0, ...)` only when (a) it has
non-empty nodes AND (b) it flows into `rationalise` / cogitation relying on
a computed signature. Leave identity `KLine(0, [])` untouched.

### Task 3 — Boundary assert in tests (option)

The `assert` at `rationalise` entry fires on misuse. Consider adding a
dedicated test in `tests/test_agent.py` that a `KLine(None, [nodes])`
(or whatever the placeholder becomes) raises on entry — documents the
contract. *(Optional; the assert itself is the enforcement.)*

## Test Mapping

| Spec ID | Test (file) | Status |
| ------- | ----------- | ------ |
| (Phase 1) | `tests/test_agent.py` — rationalise with pre-set signatures | unchanged (already pre-set) |
| (Phase 1) | `tests/test_agent.py` — rationalise with unset signature → AssertionError | new (optional) |
| (Opacity) | suite passes with `0` no longer special | verify (1254 baseline) |

## Design Decisions

| Decision | Outcome | Rationale |
| --- | --- | --- |
| Approach | remove all checks; boundary assert at entry | "let it crash" — the sentinel's two meanings are transient/structural; production never relied on lazy-derive |
| The assert | `signature is not None` (presence), not a value-test | a loud misuse crash without re-introducing a special value; `0` is ordinary |
| `expand.py:178` | delete (unreachable) | `is_identity` above already catches empty nodes |
| `stm.py:125` | delete (harmless) | empty-signature query scans and matches nothing |
| Identity klines | unchanged (`KLine(0, [])` is legitimate) | `0` is a valid signature for empty nodes; only the *unset* meaning is removed |
| Tests | fix only those relying on the derive | minimal blast radius; identity `KLine(0, [])` left alone |

## Deferred

None new. The bare-`&` overlap/gap class remains open in
`specs/signifier.md §Opacity Invariant`.

## Status

- [x] Spec (`specs/agent.md` §Phase 1, `specs/signifier.md` §Opacity Invariant)
- [x] Plan (this document)
- [x] Task 1: remove the four sentinel checks
- [x] Task 2: fix tests relying on the lazy-derive (no-op — verified no test relies on it)
- [x] Task 3: optional boundary-assert test (verified manually; assert fires on misuse)
