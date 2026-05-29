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
internally: `add` writes to STM only; `promote` moves from STM to the
frame; the base is read-only, established at construction. Lookups merge
across all tiers transparently.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### Kline (@kline spec)

- A Kline is an identified, ordered sequence of zero or more nodes.
- Signatures are uint64, not inherently unique.
- Nodes are opaque uint64 values.

### STM (@stm spec)

- STM is a bounded, dual-keyed index over recently added KLines.
- The Model manages the STM internally; callers never interact with it directly.

### Significance (@significance spec)

- Significance calls the Model's `expand` API, which yields `QueryCandidate`
  results with pre-computed significance values.
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

| Tier  | Purpose               | Bounded | Lifetime       | Written by add | Written by promote |
| ----- | --------------------- | ------- | -------------- | -------------- | ------------------ |
| STM   | Transitive grounding  | Yes     | Rolling window | Yes            | No                 |
| Frame | Session write surface | No      | Per-session    | No             | Yes (from STM)     |
| Base  | Long-term knowledge   | No      | Persistent     | No             | No                  |

### STM (Short-Term Memory)

See the **@stm spec** for the full definition. Summary:

- Bounded, dual-keyed index over recently added KLines (default bound: **256**).
- Indexes each KLine by **signature** and **nodes signature** (via `make_signature`).
- FIFO eviction when bound is exceeded; evicted KLines are discarded unless
  they have been promoted to the frame.
- Enables **transitive grounding** — finding KLines that share node structure
  even when their signatures differ.
- `add` writes to STM only.

### Frame

The frame is populated by **promotion** from STM. When the agent
determines a KLine is significant (S1 or S4), it promotes the KLine from
STM to the frame. The frame has no fixed bound — it grows as KLines are
promoted during a session.

The frame is a read-through layer over the base:

- Lookups that miss in the STM or frame fall through to the base.
- Additions to the frame (via promote) do **not** modify the base.

### Base Model

A model may optionally reference a **base model**.

- The base is **read-through**: lookups that miss in STM and frame fall
  through to the base.
- The base model is **never modified** by `add` or `promote`. It is
  established at model instantiation and remains read-only for the
  session's lifetime.
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

Adds a KLine to the model.

- Returns `true` if the KLine was added.
- Returns `false` if the KLine was rejected.
- The KLine is added to **STM only** (not the frame or base).
- If `kline.is_literal()` is `true` (all nodes are literal tokens,
  per `is_literal` in the @kline spec), a duplicate check is performed
  (see Deduplication). Non-literal Klines are always accepted.
- If adding to the STM would exceed `stm_bound`, the oldest STM entry
  is evicted. Evicted KLines that have not been promoted are discarded.

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

The total number of Klines in the frame (excluding STM and base). STM
entries that have not been promoted do not contribute to the count.

## STM Interface

These methods provide read access to the STM tier without exposing the
underlying STM object. External code must use these methods rather than
accessing the STM directly.

### STM Contains

```
model.stm_contains(kline) → bool
```

Returns whether an equal KLine exists in the STM tier only.

- Unlike `exists()`, this checks the STM tier only — not the frame or base.
- Uses the same equality semantics as `exists()`: same signature and node
  sequence.

### Iterate STM

```
model.iter_stm() → Iterator[KLine]
```

Returns an iterator over all KLines currently in the STM, in insertion
order (oldest first).

- Returns a fresh iterator on each call.
- Does not copy — callers see live insertion-order traversal.
- External code that needs to iterate STM entries (e.g., for promotion)
  must use this method.

## Promotion

Promotion moves Klines from STM to the frame, persisting them for the
session's lifetime.

### Promote

```
model.promote(kline) → bool
```

Moves a KLine from STM to the frame.

- Returns `true` if the KLine was added to the frame.
- Returns `false` if the KLine is not in STM or is already in the frame.
- The KLine remains in STM (STM eviction handles removal naturally).
- Promotion is typically triggered by the agent on significant results
  (S1 or S4), ensuring confirmed and novel knowledge persists in the
  session.

### Promote All

```
model.promote_all() → int
```

Promotes all Klines currently in STM to the frame.

- Returns the number of Klines promoted.
- Klines already in the frame are skipped.

## Deduplication

When `add` receives a literal Kline (`is_literal()` returns `true`,
meaning all nodes are literal tokens per `is_literal` in the @kline spec), the model
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
model.query_expand(kline, depth=2) → sequence of Kline
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
(@significance spec) and by cogitation (@agent spec). Their semantics are
defined here.

### QueryCandidate

```
QueryCandidate(query: Kline, candidate: Kline, significance: int)
```

A named tuple representing a single query-candidate-significance result.
Yielded by `model.expand()` for both intermediate connotations and the
terminal significance. The model computes significance internally as
`(~packed_distance) & MASK64`, where `packed_distance` encodes S2 and S3
components. Callers never see raw distance.

### Is S1

```
model.is_s1(kline) → bool
```

Determines whether a kline is structurally grounded (S1).

- `kline` — a KLine to test.
- A kline is S1 if:
  1. Its signature fully describes its nodes (canonical):
     `make_signature(kline.nodes) == kline.signature`, OR
  2. It is countersigned by another kline in the model.
- This is a stateful test: adding or removing klines changes the result.
- S1 represents a **structurally grounded kline** — one whose signature
  and nodes are fully accounted for by the model's structure.

### Expand (Significance)

```
model.expand(query, candidate, distance=0) → Iterator[QueryCandidate]
```

A generator that expands a query-candidate pair, yielding `QueryCandidate`
results for each discovered connotation and a terminal yield with the
computed significance.

- `query` — the query Kline.
- `candidate` — a candidate Kline.
- `distance` — accumulated hop distance for recursive calls (default 0).
- **Yields** intermediate `QueryCandidate` items for each discovered
  connotation (S2 and S3 indirect relationships), followed by a terminal
  `QueryCandidate` with the computed significance for the original pair.
- **Recursive**: intermediate connotations are discovered by recursively
  calling `expand()` via `yield from`. Cycle detection prevents infinite
  recursion via a visited set of `(query.signature, candidate.signature)`
  pairs.
- **Significance** — the model computes significance internally by
  packing S2 and S3 distance components into a single uint64, then
  inverting: `(~packed) & MASK64`. Higher significance means closer match.

#### Hyperparameters

- **D_MAX** — maximum distance and maximum significance value
  (`0xFFFF_FFFF_FFFF_FFFF`). Also used as the penalty for unresolvable
  mismatched nodes.
- **MASK64** — 64-bit mask for bitwise inversion (`0xFFFF_FFFF_FFFF_FFFF`).

#### Behavioral Contract

`expand()` must satisfy these properties:

1. **Single distance** — accumulated integer. S3 connotation hops biased
   by `_pack(hop_count + _S3_BIAS)`. Callers receive significance.
2. **Significance is inverted distance** — `(~distance) & MASK64`.
3. **Distance is topology-driven** — hop distances from graph topology.
4. **S2 signifies short-circuits before S3** — overlap match yields QC and
   stops chain; `s3_connotations` not populated.
5. **Connotation is always S3** — indirect bridging always uses packed distance.
6. **Ungrounded penalty is always +1** — matched but ungrounded nodes.
7. **Bidirectional** — both query and candidate mismatched nodes contribute.
8. **Quadratic packing** — `_pack(d) = d²`.
9. **Recursive expansion** — connotations yielded as `QueryCandidate`.
10. **Cycle detection** — visited signature pairs prevent infinite recursion.

The implementation algorithm and pseudocode are in
`plans/impl/model.md` §2.

#### Per-node contributions

| Node state                                                    | Contribution          | Target component  |
| ------------------------------------------------------------- | --------------------- | ----------------- |
| Mismatched, chain reaches opposing mismatch set at hop N      | +N to total_distance   | Accumulated directly  |
| Mismatched, chain reaches signature with bitwise overlap      | yields QC, +MAX_HOP   | S2 signifies candidate |
| Mismatched, chain never reaches opposing mismatch set         | +MAX_HOP              | Accumulated directly  |
| Mismatched candidate, chain bridges via connotation           | round-trip, hop = 0   | S3 packed (always)    |
| Matched + structurally grounded (is_s1)                       | 0                     | Neutral               |
| Matched + ungrounded                                          | +1                    | Accumulated directly  |

### Is Countersigned

```
model.is_countersigned(kline) → bool
```

Returns whether a kline is **countersigned** by any kline in the model.

- `kline` — a KLine.
- Returns `true` if the model contains a kline whose signature equals
  `make_signature(kline.nodes)` and whose sole node equals `kline.signature`.
- This is a structural test — it checks whether another kline vouches for
  the given kline by having the kline's nodes_signature as its identity and
  the kline's signature as its only node.
- Ratification is checked in the fast lane during rationalise() Phase 3
  (Assess), before candidates are selected.

Query = {Q: [A, B]}
Countersigner = {AB: [Q]}

Both query and countersigning signatures can represent multiple klines.
Countersigning klines are those that have one node that references the
countersigned kline.

## STM Eviction Details

STM eviction is defined in the **@stm spec**. Summary of Model-level
consequences:

STM eviction removes the oldest Kline from the STM when `add` would cause
the STM to exceed its bound.

- Eviction removes from the STM index only.
- The evicted Kline's signature and nodes signature entries are removed from
  the STM's dual-keyed index.
- Eviction does not affect the frame or base model.
- If the evicted Kline was promoted to the frame, it remains discoverable via
  frame and base lookups. If it was not promoted, it is discarded.

## Significance Semantics

The model computes significance internally via `expand()`. This section
defines the semantics of the significance computation.

### Constants

```python
D_MAX  = 0xFFFF_FFFF_FFFF_FFFF   # maximum distance and maximum significance
MASK64 = 0xFFFF_FFFF_FFFF_FFFF   # 64-bit mask for bitwise inversion
```

### Significance Inversion

```
significance = (~min(distance, D_MAX - 1)) & MASK64
```

Higher significance = closer match. The ordering is strict:
`S1 > S2 > S3 > S4` by unsigned integer comparison.

- S1: distance = 0, significance = D_MAX (all bits set)
- S4: distance = D_MAX, significance = 0 (no bits set)

### Distance Accumulation

Distance is a single accumulated integer from graph hops. For each
mismatched node in the query and candidate, the hop chain is traversed
with a three-tier priority:

1. **Exact match (S2 direct):** Node resolves via `_edge_hops()` to a node
   in the opposite mismatch set. Adds hop count directly.
2. **Signifies match (S2 loose):** Node resolves to a signature sharing bits
   with the node value. Yields a `QueryCandidate`. Node still contributes
   `MAX_HOP` to terminal distance. Short-circuits before S3 connotation.
3. **Connotation resolution (S3):** Node resolves to a signature found in
   `s3_connotations`. Hop count packed via `_pack(hop_count + _S3_BIAS)`.
4. **Unresolved:** Adds `MAX_HOP` (default 100).

Matched-but-ungrounded nodes add 1 each.

### S3 Bias and Packing

```
_S3_BIAS = 9
_pack(distance) = distance²
```

The bias ensures S3 distances moderately exceed S2 distances while
remaining in the same order of magnitude. Quadratic packing compresses
small distances together and spreads large distances apart.

### Boundaries

Three fixed boundaries classify yielded significance values:

| Boundary | Position                | Meaning                           |
| -------- | ----------------------- | --------------------------------- |
| S1\|S2   | `D_MAX - 1`             | Only exact S1 qualifies as S1     |
| S2\|S3   | `~_S2_S3_DISTANCE`      | Packed distance threshold (100)   |
| S3\|S4   | `0`                     | Only zero-significance is S4      |

Classification cascade:

```
sig >= s12 → S1
sig >= s23 → S2
sig >= s34 → S3
else       → S4
```

### Properties

1. **Inverted metric**: significance = `(~distance) & MASK64`. Higher is more
   significant.
2. **Pessimistic**: the presence of any unmatched node prevents S1.
3. **Arithmetically comparable**: S1 > S2 > S3 > S4 by unsigned comparison.
4. **Exhaustive**: every Kline with candidates is S1, S2, or S3.
5. **S1 is trivial**: all nodes match → distance = 0 → significance = D_MAX.
6. **Topology-driven**: distance accumulated from graph hops.
7. **Boundary classification**: three fixed boundaries. Raw significance
   values are never mutated.

## Test Matrix

### Storage

| ID    | Criterion                                                      | Origin ref |
| ----- | -------------------------------------------------------------- | ---------- |
| MOD-1 | Add and find: add KLine, find by signature returns it          | — |
| MOD-2 | Add returns True on success                                     | — |
| MOD-3 | Literal dedup: duplicate literal KLine rejected (returns False) | — |
| MOD-4 | Non-literal no dedup: duplicate non-literal accepted           | — |
| MOD-5 | Exists: True after add, False before                           | — |
| MOD-6 | Find returns most recent KLine when multiple share signature   | — |
| MOD-7 | Find_all: returns all KLines with given signature across tiers | — |
| MOD-8 | Find_by_nodes: returns KLine by nodes signature                | — |
| MOD-9 | Remove: removes most recent KLine with given signature         | — |
| MOD-10 | Remove never touches base model                                | — |
| MOD-11 | Len returns frame count only (excludes STM and base)          | — |

### Three-Tier Lookup

| ID     | Criterion                                           | Origin ref |
| ------ | --------------------------------------------------- | ---------- |
| MOD-12 | STM priority: KLine in STM found before frame       | — |
| MOD-13 | Frame fallback: KLine not in STM found in frame     | — |
| MOD-14 | Base fallback: KLine not in STM/frame found in base | — |
| MOD-15 | Cross-tier dedup: literal KLine in base blocks add   | — |

### Graph Traversal

| ID     | Criterion                                   | Origin ref |
| ------ | ------------------------------------------- | ---------- |
| MOD-16 | Resolve: node resolves to KLine via find    | — |
| MOD-17 | Query_expand depth 0: returns empty          | — |
| MOD-18 | Query_expand depth 2: returns direct children | — |
| MOD-19 | Query_expand cycle detection: no infinite loop | — |
| MOD-20 | Descendants: recursive node collection       | — |
| MOD-21 | Query: find + expand combined                | — |

### Promotion

| ID     | Criterion                                          | Origin ref |
| ------ | -------------------------------------------------- | ---------- |
| MOD-22 | Promote moves KLine from STM to frame              | — |
| MOD-23 | Promote without STM entry returns False             | — |
| MOD-24 | Promote_all promotes all STM KLines to frame        | — |
| MOD-25 | Promote skips already-promoted KLines               | — |

### Significance API

| ID     | Criterion                                                              | Origin ref |
| ------ | ---------------------------------------------------------------------- | ---------- |
| MOD-26 | `is_s1` canonical: `make_signature(nodes) == signature` → True          | Origin §Significance |
| MOD-27 | `is_s1` countersigned: mutual cross-reference → True                   | Origin §Significance |
| MOD-28 | `is_s1` neither: non-canonical, non-countersigned → False              | — |
| MOD-29 | `expand` all-match ungrounded: significance reflects ungrounded count   | — |
| MOD-30 | `expand` all-mismatched unresolvable: low significance                  | — |
| MOD-31 | `expand` with edge hops: connotation yields + terminal                  | — |
| MOD-32 | `expand` S2 signifies: loose match yields QC, terminal still MAX_HOP    | — |
| MOD-33 | `expand` S2 before S3: signifies short-circuits connotation recording   | — |
| MOD-34 | `expand` S3 route: S3 bias ensures S3 distances exceed S2              | — |
| MOD-35 | `expand` connotation: indirect path → S3 connotation yield + terminal   | — |
| MOD-36 | `expand` significance always in valid uint64 range `[1, D_MAX]`        | — |
| MOD-37 | `expand` bidirectional: both sides contribute connotations + terminal   | — |
| MOD-38 | `is_countersigned`: mutual node reference detected                      | — |
| MOD-39 | Not countersigned: one-way reference → False                            | — |

### Structural Grounding

| ID     | Criterion                                                                | Origin ref |
| ------ | ------------------------------------------------------------------------ | ---------- |
| MOD-40 | `promote_participating`: query + candidate promoted after ratification    | — |
| MOD-41 | `promote_participating`: S4 identity klines in STM also promoted          | — |
| MOD-42 | `promote_participating`: S2/S3 partial klines in STM promoted             | — |
| MOD-43 | `promote_participating`: already-promoted klines not re-promoted           | — |
| MOD-44 | `classify_misfit` canonical: `S == N` → (False, False)                    | — |
| MOD-45 | `classify_misfit` underfit: `S & ~N != 0` → (True, False)                 | — |
| MOD-46 | `classify_misfit` overfit: `N & ~S != 0` → (False, True)                  | — |
| MOD-47 | `classify_misfit` dual: both conditions → (True, True)                     | — |
| MOD-48 | `generate_expansions` underfit: returns proposal with added nodes          | — |
| MOD-49 | `generate_expansions` overfit: returns trimmed + companion                 | — |
| MOD-50 | `generate_expansions` dual: returns replacement + companion                | — |
| MOD-51 | `generate_expansions` no gap: no expansion proposals emitted               | — |

## What a Model is Not

The following are explicitly **out of scope** for this spec:

- **Significance computation.** The model computes significance internally
  via `expand()`. Routing is defined in the significance spec.
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

- **Significance** (@significance spec) — calls `is_s1`, `expand`.
- **Agent** (@agent spec) — stores encoded Klines, retrieves candidates,
  traverses the graph, promotes significant Klines.
