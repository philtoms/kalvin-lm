# Model Specification

## Overview

A Model is a mutable, indexed collection of Klines with a three-tier
layered memory architecture. It provides storage, deduplication, lookup
by signature, graph traversal, and the comparison functions consumed by
the significance pipeline.

The three tiers from newest to oldest are:

```
STM → Frame → Base
```

Callers see a single unified Model API. The model manages the tiers
internally, routing adds to the appropriate tier and merging lookups
across all tiers transparently.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### Kline (@kline spec)

- A Kline is an identified, ordered sequence of zero or more nodes.
- Signatures are uint64, not inherently unique.
- Nodes are opaque uint64 values.

### Significance (@significance spec)

- Significance calls three Model API functions: `is_s1`, `s2_distance`,
  `s3_distance`.
- Significance does not manage model state.

### Agent (@agent spec)

- The agent calls model operations (`add`, `find`, `exists`, `query`, etc.)
  without knowledge of internal tiering.
- The agent is responsible for calling model operations; the model is
  responsible for managing its internal memory tiers.

## Definition

A Model consists of:

| Component | Type                   | Description                         |
| --------- | ---------------------- | ----------------------------------- |
| stm       | STM                    | Short-term memory. Bounded, recent. |
| frame     | Frame                  | Per-session write layer over base.  |
| base      | Model \| none          | Optional long-term knowledge store. |
| index     | signature → [Kline, …] | Unified index across all tiers.     |

### Tier Summary

| Tier  | Purpose               | Bounded | Lifetime       | Modified by add |
| ----- | --------------------- | ------- | -------------- | --------------- |
| STM   | Transitive grounding  | Yes     | Rolling window | Yes             |
| Frame | Session write surface | No      | Per-session    | Yes             |
| Base  | Long-term knowledge   | No      | Persistent     | No (promotion)  |

### STM (Short-Term Memory)

STM is a bounded, dual-keyed index over the most recently added Klines.
It retains only the N most recently added Klines (default bound: **256**).
When the bound is exceeded, the oldest entries are evicted.

STM indexes each Kline by two keys:

1. **signature** — `kline.signature`
2. **nodes signature** — the OR-reduction of all nodes in `kline.nodes`,
   equivalent to `make_signature(kline.nodes)` as defined in the @signature
   spec.

Both keys map to the same Kline. When the two keys are identical the Kline
is stored under a single key.

STM enables **transitive grounding**: when a Kline is added, its
nodes signature is indexed alongside its signature. A subsequent lookup
by nodes signature retrieves Klines that share the same node structure,
providing grounding evidence even when the signatures differ.

The STM bound is configurable at construction time.

### Frame

The frame is the primary write surface for the current session. All
non-rejected Klines are added to the frame. The frame has no fixed bound
— it grows as Klines are added during a session.

The frame is a read-through layer over the base:

- Lookups that miss in the STM or frame fall through to the base.
- Additions to the frame do **not** modify the base.

### Base Model

A model may optionally reference a **base model**.

- The base is **read-through**: lookups that miss in STM and frame fall
  through to the base.
- The base model is **not directly modified** by `add`. Klines are
  promoted to the base via a separate mechanism (see Promotion).
- A user request instantiates a new model layered over a shared base, so
  each session has an isolated write surface with access to shared
  knowledge.

### Index

The index is a unified, derived structure maintained across all tiers:

- Every Kline in any tier is indexed by its `signature`.
- Multiple Klines may share the same signature (signatures are not unique).
- The index supports O(1) lookup by signature.
- The index is maintained by the model internally — callers never interact
  with it directly.

## Construction

```
Model(base=None, stm_bound=256)
```

- `base` — optional, an existing Model serving as the long-term store.
  Defaults to none.
- `stm_bound` — maximum number of Klines retained in STM. Defaults to 256.
- A newly constructed model contains zero Klines in all tiers.

## Lookup Semantics

All read operations (`find`, `find_all`, `exists`, `resolve`, `where`,
`query`) search tiers in order:

```
STM → Frame → Base
```

A Kline found in any tier is returned. If the same signature exists in
multiple tiers, the most recently added Kline is returned (STM has
priority, then frame, then base).

## Storage Operations

### Add

```
model.add(kline) → bool
```

Adds a Kline to the model.

- Returns `true` if the Kline was added.
- Returns `false` if the Kline was rejected.
- The Kline is added to **both** the STM and the frame.
- If `kline.is_literal()` is `true` (all nodes are literal tokens,
  per @tokenizer spec), a duplicate check is performed
  (see Deduplication). Non-literal Klines are always accepted.
- If adding to the STM would exceed `stm_bound`, the oldest STM entry
  is evicted. The evicted Kline remains in the frame.
- The base model is **not modified** by `add`.

### Exists

```
model.exists(kline) → bool
```

Returns whether an equal Kline is already stored in any tier.

- Two Klines are equal when their signatures and node sequences are equal
  (kline equality, @kline spec).
- Searches STM, then frame, then base.

### Find

```
model.find(signature) → Kline | none
```

Returns a Kline by signature.

- If multiple Klines share the signature across tiers, returns the
  **most recently added** (STM first, then frame, then base).
- Returns `none` if no Kline with that signature exists in any tier.

### Find by Nodes Signature

```
model.find_by_nodes(nodes_signature) → Kline | none
```

Returns the most recently added Kline whose nodes signature matches.

- The nodes signature is the OR-reduction of all nodes in a Kline's node
  sequence, equivalent to `make_signature(kline.nodes)` as defined in the
  @signature spec.
- Searches STM first (primary index for nodes signatures), then frame,
  then base.
- Returns `none` if no Kline with that nodes signature exists.

### Find All

```
model.find_all(signature) → sequence of Kline
```

Returns all Klines with the given signature across all tiers.

- Returns Klines in insertion order (oldest first).
- STM results, frame results, and base results are merged.

### Remove

```
model.remove(signature) → bool
```

Removes the most recently added Kline with the given signature.

- Returns `true` if a Kline was removed.
- Returns `false` if no Kline with that signature exists.
- Removal applies to the tier where the most recently added Kline resides.
- Removal never affects the base model.

### Count

```
len(model) → int ≥ 0
```

The total number of Klines in the frame (excluding STM and base). STM is
a rolling window over the frame's Klines and does not contribute
independently to the count.

## Promotion

Promotion moves Klines from the frame to the base model, making them
persistent across sessions.

### Promote

```
model.promote(kline) → bool
```

Adds a Kline directly to the base model.

- Returns `true` if the Kline was added to the base.
- Returns `false` if the Kline was rejected (e.g., duplicate in base).
- The Kline remains in the frame and STM.
- Promotion is typically triggered by the agent on significant results
  (S1 or S4), ensuring confirmed and novel knowledge propagates to the
  long-term store.

### Promote All

```
model.promote_all() → int
```

Promotes all Klines in the frame to the base model.

- Returns the number of Klines promoted.
- Klines that duplicate entries in the base are skipped.
- Useful for session-end persistence.

## Deduplication

When `add` receives a literal Kline (`is_literal()` returns `true`,
meaning all nodes are literal tokens per @tokenizer spec), the model
checks whether an equal Kline (same signature, same node sequence per
kline equality) already exists in **any tier**. If so, `add` returns
`false` and no entry is created.

Non-literal Klines are always accepted — composed structures are never
deduplicated.

Deduplication is cross-tier: a literal Kline that duplicates an entry in
the base model is rejected. To force-add a Kline that shadows the base,
use `add` on a model without that base, or accept the deduplication
behaviour.

## Iteration

### All Klines

```
model.klines() → sequence of Kline
```

Returns all Klines in reverse insertion order (most recent first).

- Includes Klines from all tiers: STM entries first (most recent), then
  frame entries not in STM, then base entries not in frame.
- Duplicates across tiers are suppressed: each unique Kline appears once.

### Filtered Iteration

```
model.where(predicate) → sequence of Kline
```

Returns Klines matching a predicate, in reverse insertion order.

- `predicate` — a function `(Kline) → bool`, or an int (KSig) for AND matching.
- Searches all tiers. Duplicates suppressed.
- The model does not define what predicates are valid. That is caller-defined.

## Graph Traversal

A Kline's nodes are uint64 values. When a node value equals the signature of
another Kline in the model, it forms an **edge** in the knowledge graph. Graph
traversal resolves these edges.

All graph traversal operations search across all tiers.

### Resolve

```
model.resolve(node) → Kline | none
```

Resolves a node value to the Kline whose signature matches, if one exists.

- Equivalent to `model.find(node)`.
- Returns `none` if no Kline has that signature in any tier.

### Expand

```
model.expand(kline, depth=2) → sequence of Kline
```

Traverses the graph starting from `kline`, resolving each node to a child
Kline, then recursing into children up to `depth` levels.

- `depth=0` — returns empty.
- `depth=1` — returns empty (the starting kline itself is not included; only
  resolved children and their descendants).
- `depth=2` — resolves the kline's nodes to direct children and yields them.
- `depth=N` — resolves children, then their children, up to N − 1 levels
  deep.
- **Cycle detection**: a Kline visited during expansion is not visited again.
- **Missing nodes**: a node that does not resolve to any Kline is skipped.
- **Order**: children are yielded in node order, depth-first.

### Descendants

```
model.descendants(node) → set of uint64
```

Recursively collects all descendant node values starting from the Kline
identified by `node`.

- Resolves `node` to a Kline, then traverses all children recursively.
- Returns a **set** of uint64 node values (no duplicates, no order).
- **Cycle detection**: a node visited is not re-traversed.
- Returns an empty set if `node` does not resolve.

### Query

```
model.query(signature, depth=1) → sequence of Kline
```

Returns all Klines whose signature equals `signature`, then expands each
match to `depth`.

- First finds all Klines with the given signature across all tiers.
- Then expands each match using the same semantics as `expand`.
- Yields results in reverse insertion order for matches, depth-first for
  expansions.

## Model API (Significance)

The following functions are consumed by the significance pipeline
(@significance spec) and by cogitation countersignature discovery
(@agent spec). Their semantics are defined here; their implementation
is TBD.

### Is S1

```
model.is_s1(node, candidate) → bool
```

Returns whether a single node achieves S1 significance against a candidate
Kline.

- `node` — a uint64 value from the query Kline's node sequence.
- `candidate` — a Kline from the model.
- S1 represents a **perfect match** at that position.
- Semantics of the match test are **TBD**.

### S2 Distance

```
model.s2_distance(query, candidate) → uint64
```

Returns a distance value when some (but not all) nodes achieve S1 against
the candidate.

- `query` — the query Kline.
- `candidate` — a candidate Kline.
- Must return a value in `[1, D_boundary)`.
- Values outside range are clamped.
- Semantics are **TBD**.

### S3 Distance

```
model.s3_distance(query, candidate) → uint64
```

Returns a distance value when no nodes achieve S1 against the candidate.

- `query` — the query Kline.
- `candidate` — a candidate Kline.
- Must return a value in `[D_boundary, 0xFFFF_FFFF_FFFF_FFFF)`.
- Values outside range are clamped.
- `D_boundary` is a hyperparameter (default `0x8000_0000_0000_0000`).
- Semantics are **TBD**.

### Is Countersigned

```
model.is_countersigned(A, B) → bool
```

Returns whether two Klines are **countersigned** — each references the
other through its nodes.

- `A`, `B` — two Klines.
- Returns `true` if `B.signature` appears in `A.nodes` AND
  `A.signature` appears in `B.nodes`.
- This is a structural test on node values, not a significance computation.
- Literal nodes cannot match a signature (literal tokens use a 32-bit mask
  in the lower bits, which does not equal any signature value), so the test
  naturally considers only non-literal matches — but this is enforced by
  the encoding, not by an explicit filter.
- Countersignature is a latent relationship typically discovered during
  cogitation, not during initial rationalisation.

This function is composed from existing model primitives (`find`, `resolve`)
and is provided as a convenience for the cogitation pipeline.

## STM Eviction Details

STM eviction removes the oldest Kline from the STM when `add` would cause
the STM to exceed its bound.

- Eviction removes from the STM index only. The Kline remains in the frame.
- The evicted Kline's signature and nodes signature entries are removed from
  the STM's dual-keyed index.
- Eviction does not affect the frame or base model.
- After eviction, the evicted Kline is still discoverable via frame and base
  lookups, but no longer via STM-specific nodes-signature indexing.

## What a Model is Not

The following are explicitly **out of scope** for this spec:

- **Significance computation.** The model provides raw comparison functions;
  routing, distance-to-significance conversion, and level assignment are
  defined in the significance spec.
- **Encoding.** Converting text or other input into Klines is the agent's
  responsibility (@agent spec).
- **Tokenisation.** Producing nodes from input is defined in the
  @tokenizer spec. Producing signatures from nodes is defined in the
  @signature spec.
- **Persistence.** Serialisation and deserialisation are implementation-level
  concerns.
- **Debug metadata.** Labels, source text, timestamps, or other diagnostic
  data attached to entries.
- **Thread management.** How concurrent access to tiers is managed is an
  implementation concern.

## Referenced By

- **Significance** (@significance spec) — calls `is_s1`, `s2_distance`,
  `s3_distance`.
- **Agent** (@agent spec) — stores encoded Klines, retrieves candidates,
  traverses the graph, promotes significant Klines.
