# Expand Robustness Specification

## Overview

The graph expansion pipeline (`expand()`, `edge_hops()`) must be robust against
cycles, identity klines, and unresolvable signatures. It must never raise
exceptions due to graph topology.

## Dependencies

### Model (@model spec)

- `model.find(sig)` returns a KLine or None.
- `model.where(predicate)` returns matching klines.

### Signature (@signature spec)

- `make_signature(nodes)` — OR-reduction. `make_signature([]) = 0`.

### Kline (@kline spec)

- Identity kline: signature > 0, nodes = [].

## Definition

### edge_hops

`edge_hops(model, sig)` yields `(hop_count, next_sig)` for each non-canonical
resolution step in the chain `sig → kline → make_signature(kline.nodes) → ...`.

#### Behavioural Rules

**ER-1: Cycle detection.** If the chain revisits a signature already seen in
the current traversal, the function breaks immediately without yielding the
cycle. This prevents infinite oscillation between countersigned klines
(e.g., `{M: [H]}` ↔ `{H: [M]}`) from consuming all MAX_HOP iterations.

**ER-2: Identity kline guard.** If `make_signature(kline.nodes) == 0` (identity
kline with empty nodes), the function breaks without yielding. The resolved
chain has reached a dead end — there is no signature to follow from `nodes = []`.

**ER-3: Canonical termination.** If the kline is canonical
(`sig == make_signature(nodes)`), the function breaks. Unchanged from prior
behaviour.

**ER-4: Dead-end termination.** If `model.find(sig)` returns None, the function
breaks. Unchanged from prior behaviour.

**ER-5: MAX_HOP bound.** The total number of yields never exceeds MAX_HOP.
Unchanged from prior behaviour.

### expand

`expand(model, query, candidate, distance, _visited)` yields `QueryCandidate`
objects for each discovered relationship.

#### Behavioural Rules

**ER-6: Null-safe resolution.** When `edge_hops` yields a `match_sig` that is
used for recursive expansion, `expand` must resolve it via `model.find()`
rather than asserting it exists. If `model.find(match_sig)` returns None, the
recursive expansion is silently skipped — no exception is raised.

This applies to all three expansion call sites:
- Mismatched query node reaches a mismatched candidate node
- Mismatched candidate node reaches a mismatched query node
- S3 connotation bridging

**ER-7: Graceful degradation.** Unresolvable match signatures are not an error
condition — they represent graph topology where a resolution chain leads to a
signature not present in the model. The expansion continues processing
remaining yields.

## Out of Scope

- Significance normalization (see @significance-normalization spec).
- Expansion proposal generation (`propose_expansions`).
- Changes to the `expand` distance computation or boundary classification.

## Test Matrix

| ID | Criterion | Category |
|----|-----------|----------|
| ER-1 | Countersigned pair (M↔H) yields at most 2 hops, not MAX_HOP | edge_hops |
| ER-2 | Identity kline `{A: []}` yields zero hops | edge_hops |
| ER-3 | Canonical kline yields zero hops (unchanged) | edge_hops |
| ER-4 | Unresolvable sig yields zero hops (unchanged) | edge_hops |
| ER-5 | Total yields never exceed MAX_HOP (unchanged) | edge_hops |
| ER-6 | expand() does not crash when match_sig is unresolvable | expand |
| ER-7 | S2 expansion scenario completes without exception | expand |
