# Model Specification

## Overview

A Model is a mutable, indexed collection of Klines. It provides storage,
deduplication, lookup by signature, graph traversal, and the comparison
functions consumed by the significance pipeline.

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

## Definition

A Model consists of:

| Component   | Type                          | Description                              |
|-------------|-------------------------------|------------------------------------------|
| entries     | ordered collection of Kline   | All Klines in insertion order.           |
| index       | signature → [Kline, …]       | Groups Klines by signature. O(1) lookup. |
| base        | Model \| none                 | Optional read-through base model.        |

### Entries

- Klines are stored in insertion order.
- Iteration follows **reverse insertion order** (most recently added first).
- An entry is a Kline exactly as defined in the kline spec — no additional
  fields, no hidden metadata.

### Index

- Every Kline is indexed by its `signature`.
- Multiple Klines may share the same signature (signatures are not unique).
- The index is derived state, maintained by the model internally.

### Base Model

- A model may optionally reference a **base model**.
- The base is **read-through**: lookups that miss in the current model fall
  through to the base.
- Adds, updates, and deduplication apply only to the current model.
- A user request instantiates a new model layered over a shared base, so
  each request has an isolated write surface with access to shared knowledge.
- The base model is **not modified** by operations on the derived model.

## Construction

```
Model(base=None)
```

- `base` — optional, an existing Model. Defaults to none (empty model).
- A newly constructed model contains zero Klines.

## Storage Operations

### Add

```
model.add(kline) → bool
```

Adds a Kline to the model.

- Returns `true` if the Kline was added.
- Returns `false` if the Kline was rejected.
- The Kline is appended to `entries` and indexed by its signature.
- If `kline.is_literal()` is `true`, a duplicate check is performed
  (see Deduplication). Non-literal Klines are always accepted.
- If a base model is present, the added Kline does **not** modify the base.

### Exists

```
model.exists(kline) → bool
```

Returns whether an equal Kline is already stored.

- Two Klines are equal when their signatures and node sequences are equal
  (kline equality, @kline spec).
- If not found in the current model and a base exists, checks the base.

### Find

```
model.find(signature) → Kline | none
```

Returns a Kline by signature.

- If multiple Klines share the signature, returns the **most recently added**.
- If not found in the current model and a base exists, queries the base.
- Returns `none` if no Kline with that signature exists in either layer.

### Find All

```
model.find_all(signature) → sequence of Kline
```

Returns all Klines with the given signature.

- Returns Klines in insertion order (oldest first).
- If a base exists, base results are included after current-model results.

### Remove

```
model.remove(signature) → bool
```

Removes the most recently added Kline with the given signature.

- Returns `true` if a Kline was removed.
- Returns `false` if no Kline with that signature exists.
- Removal never affects the base model.

### Count

```
len(model) → int ≥ 0
```

The number of Klines in the current model (excluding the base).

## Deduplication

When `add` receives a literal Kline (`is_literal()` returns `true`), the
model checks whether an equal Kline (same signature, same node sequence per
kline equality) already exists. If so, `add` returns `false` and no entry
is created.

Non-literal Klines are always accepted — composed structures are never
deduplicated.

Deduplication applies only within the current model. A literal Kline that
duplicates an entry in the base model **is** added to the current model
(the current model's copy shadows the base).

## Iteration

### All Klines

```
model.klines() → sequence of Kline
```

Returns all Klines in reverse insertion order (most recent first).

- Includes only the current model's Klines, not the base.

### Filtered Iteration

```
model.where(predicate) → sequence of Kline
```

Returns Klines matching a predicate, in reverse insertion order.

- `predicate` — a function `(Kline) → bool`.
- The model does not define what predicates are valid. That is caller-defined.

## Graph Traversal

A Kline's nodes are uint64 values. When a node value equals the signature of
another Kline in the model, it forms an **edge** in the knowledge graph. Graph
traversal resolves these edges.

### Resolve

```
model.resolve(node) → Kline | none
```

Resolves a node value to the Kline whose signature matches, if one exists.

- Equivalent to `model.find(node)`.
- Returns `none` if no Kline has that signature.
- Includes the base model in lookup.

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
- Includes the base model in resolution.

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
- Includes the base model in resolution.

### Query

```
model.query(signature, depth=1) → sequence of Kline
```

Returns all Klines whose signature equals `signature`, then expands each
match to `depth`.

- First finds all Klines with the given signature (including base).
- Then expands each match using the same semantics as `expand`.
- Yields results in reverse insertion order for matches, depth-first for
  expansions.

## Model API (Significance)

The following functions are consumed by the significance pipeline
(@significance spec). Their semantics are defined here; their implementation
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

## What a Model is Not

The following are explicitly **out of scope** for this spec:

- **Significance computation.** The model provides raw comparison functions;
  routing, distance-to-significance conversion, and level assignment are
  defined in the significance spec.
- **Encoding.** Converting text or other input into Klines is the agent's
  responsibility (@agent spec).
- **Tokenisation.** Producing signatures from input is defined in the
  tokeniser spec.
- **Persistence.** Serialisation and deserialisation are implementation-level
  concerns.
- **Debug metadata.** Labels, source text, timestamps, or other diagnostic
  data attached to entries.

## Referenced By

- **Significance** (@significance spec) — calls `is_s1`, `s2_distance`,
  `s3_distance`.
- **Agent** (@agent spec) — stores encoded Klines, retrieves candidates,
  traverses the graph.
