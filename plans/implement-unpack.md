# Unpack Implementation Plan

**Parent:** kline decode correctness (passing kline nodes directly to the
tokenizer does not account for packed/multi-token nodes)
**Status:** complete (all tasks landed — see §Status)
**Spec refs:** `specs/model.md` §Graph Traversal › Unpack (MOD-60..MOD-66)
**Depends on:** —

## Problem

A kline node value may be a **packed signature** — the OR-reduction of two or
more token IDs (a §11.4 multi-token word, a §11.5 compound). Passing such a
value directly to a tokenizer's `decode` is incorrect: the low 32 bits are
not a single valid token ID. Decode of node/signature values must first
flatten the value to its constituent identity signatures via the graph, then
hand that sequence to the tokenizer. This plan adds that flattening
operation; tokenizer integration is left to callers (out of scope — see
Design Decisions).

## Spec References

- `@specs/model.md` §Graph Traversal › Unpack — MOD-60..MOD-66
- `@specs/model.md` §Removed Methods — MOD-R4
- `@specs/kline.md` — node order significance, identity
- `@specs/signature.md` — `make_signature`
- `@CONTEXT.md` §Identity, §Recency Precedence

## Implementation Tasks

### Task 1 — `Model.unpack(kline)` (`src/kalvin/model.py`)

- **Spec ref:** @specs/model.md §Graph Traversal › Unpack — MOD-60..MOD-66
- **Pseudocode:**
  ```
  def unpack(self, kline):
      # base case: identity
      if not kline.nodes:
          return [kline.signature]
      # canon: signature == make_signature(nodes)
      if kline.signature == make_signature(kline.nodes):
          out = []
          for node in kline.nodes:
              child = self._resolve_for_unpack(node)   # kind precedence
              out.extend(self.unpack(child))
          return out
      raise <non-decomposable input>
  ```
- `_resolve_for_unpack(node)`:
  - candidates = `find_all(node)` (ordered most-recent-first by tier chain)
  - filter to identities (empty nodes) first; if any, return the
    most-recently-added identity
  - else filter to canons (`signature == make_signature(nodes)`); if any,
    return the most-recently-added canon
  - else raise
  - **Note:** `find_all` returns insertion order (oldest first); the most
    recently added is the *last* element within a tier group. Verify the
    exact ordering against `_TierChain.find_all` before relying on it.
- No `visited` set / cycle detection (by decision — see Design Decisions).

### Task 2 — Remove dead code

- `src/kalvin/expand.py`: remove `_pack` (lines ~44–54) and its mention in
  the module docstring re-export list (line ~19).
- `src/kalvin/model.py`: remove `descendants`, `_descendants_inner`,
  `get_all_descendants`.
- `tests/test_model.py`: remove `test_descendants` and `test_cycle_detection`
  (both exercise the removed `descendants`).

### Task 3 — Tests (`tests/test_model.py`)

Add a `TestUnpack` group covering MOD-60..MOD-66:

| Spec ID | Test                                   |
| ------- | -------------------------------------- |
| MOD-60  | identity kline → `[signature]`         |
| MOD-61  | canon → ordered identity children      |
| MOD-62  | canon-of-canons → flattened, ordered   |
| MOD-63  | connoted input (not identity/canon) raises |
| MOD-64  | canon with a node resolving to no kline raises |
| MOD-65  | node heads both an identity and a connoted pair → identity wins (loop prevention) |
| MOD-66  | two canons share a signature → most-recently-added nodes used |

## Design Decisions

Resolved during the grilling session that produced this plan.

| Decision | Outcome | Rationale |
| --- | --- | --- |
| Input unit | a kline; recursion walks its node tree | only identity signatures decode to text |
| Base case | identity (empty nodes) → `[signature]` | the identity's single token is the terminal |
| Recursive case | canon (`signature == make_signature(nodes)`) → concat `unpack(child)` per node | only canon has a recoverable decomposition |
| Kind precedence | identity → canon → raise | semantic-relation klines contribute no tokens; following them risks non-termination (e.g. countersigned pairs) |
| Within-kind ambiguity | most-recently-added wins (Recency Precedence) | core Kalvin principle; now grounded in CONTEXT.md |
| Unresolvable node | raise | treat missing data as a bug at this stage |
| Non-decomposable input | raise | S2-misfit / S3 traversal deferred |
| Cycle defense | none — let it overflow | trust the DAG invariant; cycles are malformed-graph bugs |
| Return type | `list[int]` (identity signature values) | the natural input to a tokenizer `decode` |
| Placement | `Model.unpack(kline)` method | recursive node-tree walk is the same shape as the (now-removed) `descendants`; no import cycle (model already imports `make_signature`) |
| Tokenizer | out of scope — `unpack` returns the sequence only | keeps the model-dependent and tokenizer-dependent concerns separate; callers compose `tokenizer.decode(model.unpack(...))` |
| Naming | `unpack`, not `decode` | it flattens the node tree; decoding happens at the tokenizer |
| `kline_display` | untouched | it is a throwaway helper; a future decompiler may replace it |

## Deferred (side note)

unpack currently handles identities and clean S2 canons only. Three
bit-overlap regimes carry node information the decoder must eventually walk:
S2 canon (full overlap), S2 misfit (partial overlap), and S3 connoted /
undersigned (no overlap). Folding S2-misfit and S3 into the resolution will
likely require reading structural state rather than the identity-vs-canon
test. Separate session.

## Status

- [x] Spec (`specs/model.md` §Unpack, MOD-60..66, MOD-R4)
- [x] Plan (this document)
- [x] Task 1: `Model.unpack`
- [x] Task 2: dead-code removal
- [x] Task 3: tests
