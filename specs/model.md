# Model Specification

## Overview

A Model is a mutable, indexed collection of Klines with a four-tier
layered memory architecture. It provides storage, deduplication, lookup
by signature, graph traversal, and the comparison functions consumed by
the significance pipeline.

The four tiers from newest to oldest are:

```
STM → Frame → LTM → Base
```

Callers see a single unified Model API with three write methods forming a
cascade: `add_to_stm()` writes to STM only; `add_to_frame()` writes to Frame and
cascades to STM; `add_to_ltm()` writes to LTM and cascades through Frame to
STM. The caller selects the appropriate entry point based on the
significance outcome of rationalisation. Base is read-only, established
at construction. Lookups merge across all tiers transparently.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### Kline (@kline spec)

- A Kline is an identified, ordered sequence of zero or more nodes.
- Signatures are uint64, not inherently unique.
- Nodes are opaque uint64 values.

### STM (@stm spec)

- STM is a bounded, dual-keyed index over recently added KLines.
- The Model manages the STM internally; callers never interact with it directly.

### Frame (@CONTEXT.md §Frame)

- Frame is compressed working context — only recognised klines reach the Frame.
- Written via `add_to_frame()`, which cascades to STM.
- Monotonic — klines are appended, never removed.

### LTM (@CONTEXT.md §LTM)

- LTM (Long-Term Memory) holds ratified (S1) and novel (S4) klines.
- Written via `add_to_ltm()`, which cascades through Frame to STM.
- Structurally identical to Frame — the distinction is semantic.
- Persisted across sessions, loaded at session start. Monotonic.

### Significance (@significance spec)

- Significance calls the Model's `expand` API, which yields `QueryCandidate`
  results with pre-computed significance values.
- Significance does not manage model state.

### Agent (@agent spec)

- The agent calls model write operations (`add_to_stm`, `add_to_frame`, `add_to_ltm`)
  and read operations (`find`, `exists`, `query`, etc.).
- The agent selects the appropriate write method based on the significance
  outcome of rationalisation.
- The model is responsible for managing its internal memory tiers and the
  cascade semantics of write operations.

## Definition

A Model consists of:

| Component | Type                   | Description                                 |
| --------- | ---------------------- | ------------------------------------------- |
| stm       | STM                    | Short-term memory. Bounded, recent.         |
| frame     | Frame                  | Working context. Populated by `add()`.      |
| ltm       | LTM                    | Long-term memory. Populated by `promote()`. |
| base      | Model \| none          | Optional read-only knowledge store.         |
| index     | signature → [Kline, …] | Unified index across all tiers.             |

### Tier Summary

| Tier  | Purpose              | Bounded | Lifetime       | add_to_stm | add_to_frame | add_to_ltm |
| ----- | -------------------- | ------- | -------------- | ---------- | ------------ | ---------- |
| STM   | Event register       | Yes     | Rolling window | Yes        | Yes          | Yes        |
| Frame | Compressed context   | No      | Per-session    | No         | Yes          | Yes        |
| LTM   | Persistent knowledge | No      | Cross-session  | No         | No           | Yes        |
| Base  | Long-term knowledge  | No      | Persistent     | No         | No           | No         |

### STM (Short-Term Memory)

See the **@stm spec** for the full definition. Summary:

- Bounded, dual-keyed index over recently added KLines (default bound: **256**).
- Indexes each KLine by **signature** and **nodes signature** (via `make_signature`).
- FIFO eviction when bound is exceeded; evicted KLines remain in Frame and
  deeper tiers.
- Enables **transitive grounding** — finding KLines that share node structure
  even when their signatures differ.
- `add_to_stm()` always refreshes FIFO position (removes-if-present then adds).
- STM is populated via the write cascade: `add_to_ltm()` → Frame → `add_to_frame()` → STM → `add_to_stm()`.

### Frame

The frame is populated by `add_to_frame()` and as a cascade from `add_to_ltm()`.
Not all klines reach the Frame — only those from significant outcomes
(expanded proposals, ratified klines, novel klines). The incoming query
kline on the slow path goes to STM only. The frame has no fixed bound — it
grows as KLines are written during a session. Frame is monotonic.

Lookups that miss in STM fall through to Frame, then LTM, then Base.
KLines in Frame take precedence over the same kline in LTM or Base.

### LTM (Long-Term Memory)

LTM holds ratified (S1) and novel (S4) klines. Structurally identical to
Frame — the distinction is semantic. Populated by `add_to_ltm()`, which
cascades through Frame to STM. LTM is monotonic.

- LTM is loaded at session start from the previous session's persisted state.
- LTM is persisted at session end alongside Frame (separate sections, never merged).
- Lookups that miss in Frame fall through to LTM.
- `add_to_ltm()` cascades: writes LTM, then Frame, then STM. All three tiers
  receive the kline.

### Base Model

A model may optionally reference a **base model**.

- The base is **read-through**: lookups that miss in STM, Frame, and LTM
  fall through to the base.
- The base model is **never modified** by `add` or `promote`. It is
  established at model instantiation and remains read-only for the
  session's lifetime.

### Cross-Tier Search (\_TierChain)

Cross-tier lookups across STM, Frame, LTM, and Base are centralised in
the private `_TierChain` and `_TierAdapter` classes (internal to
`model.py`). `_TierAdapter` normalises the heterogeneous tier interfaces
(STM, `KLineStore`, and `Model`) into a uniform read surface.
`_TierChain` holds an ordered list of adapters and delegates the five
cascade methods (`find`, `find_all`, `find_by_nodes`, `_exists_any`,
`klines`) to them. Adding a new tier requires editing one place instead
of five.

### Index

The index is a unified, derived structure maintained across all tiers:

- Every Kline in any tier is indexed by its `signature`.
- Multiple Klines may share the same signature (signatures are not unique).
- The index supports O(1) lookup by signature.
- The index is maintained by the model internally — callers never interact
  with it directly.

## Construction

```
Model(base=None, ltm=None, stm_bound=256)
```

- `base` — optional, an existing Model serving as the read-only knowledge
  store. Defaults to none.
- `ltm` — optional, an existing Model serving as the long-term memory.
  Loaded from the previous session's persisted LTM. Defaults to none.
- `stm_bound` — maximum number of Klines retained in STM. Defaults to 256.
- A newly constructed model contains zero Klines in STM and Frame. LTM and
  Base are populated from their respective parameters.

## Lookup Semantics

All read operations (`find`, `find_all`, `exists`, `resolve`, `where`,
`query`) search tiers in order:

```
STM → Frame → LTM → Base
```

A Kline found in any tier is returned. If the same signature exists in
multiple tiers, the most recently added Kline is returned (STM has
priority, then Frame, then LTM, then Base).

## Storage Operations

The model provides three write methods forming a cascade:

```
add_to_ltm()  →  writes LTM  →  calls add_to_frame()
add_to_frame()  →  writes Frame  →  calls add_to_stm()
add_to_stm()  →  writes STM only (always refreshes FIFO)
```

All three methods return void. All writes are unconditional (no early-return
guards).

### Add to STM

```
model.add_to_stm(kline) → None
```

Writes a KLine to STM only.

- Always refreshes FIFO position: removes the kline from STM if present,
  then adds it fresh.
- If adding would exceed `stm_bound`, the oldest STM entry is evicted.
  Evicted KLines remain in Frame and deeper tiers.
- Absorbs the old `refresh_stm()` method — always refreshes unconditionally.

### Add to Frame

```
model.add_to_frame(kline) → None
```

Writes a KLine to Frame, then cascades to `add_to_stm()`.

- Frame is monotonic — klines are appended, never removed.
- The KLineStore membership index (`_dedup`) provides fast `contains()`
  checks.
- Cascades to `add_to_stm()`, which refreshes STM FIFO position.

### Add to LTM

```
model.add_to_ltm(kline) → None
```

Writes a KLine to LTM, then cascades to `add_to_frame()`.

- LTM is monotonic — klines are appended, never removed.
- The KLineStore membership index (`_dedup`) provides fast `contains()`
  checks.
- Cascades to `add_to_frame()`, which cascades to `add_to_stm()`.

### Exists

```
model.exists(kline) → bool
```

Returns whether an equal Kline is already stored in any tier.

- Two Klines are equal when their signatures and node sequences are equal
  (kline equality, @kline spec).
- Searches STM, then Frame, then LTM, then Base.

### Find

```
model.find(signature) → Kline | none
```

Returns a Kline by signature.

- If multiple Klines share the signature across tiers, returns the
  **most recently added** (STM first, then Frame, then LTM, then Base).
- Returns `none` if no Kline with that signature exists in any tier.

### Find by Nodes Signature

```
model.find_by_nodes(nodes_signature) → Kline | none
```

Returns the most recently added Kline whose nodes signature matches.

- The nodes signature is the OR-reduction of all nodes in a Kline's node
  sequence, equivalent to `make_signature(kline.nodes)` as defined in the
  @signature spec.
- Searches STM first (primary index for nodes signatures), then Frame,
  then LTM, then Base.
- Returns `none` if no Kline with that nodes signature exists.

### Find All

```
model.find_all(signature) → sequence of Kline
```

Returns all Klines with the given signature across all tiers.

- Returns Klines in insertion order (oldest first).
- STM results, Frame results, LTM results, and Base results are merged.

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

The total number of Klines in the Frame (excluding STM, LTM, and Base).
STM entries and LTM entries do not contribute to the count.

## STM Read Interface

These methods provide read access to the STM tier without exposing the
underlying STM object. External code must use these methods rather than
accessing the STM directly.

### STM Contains

```
model.stm_contains(kline) → bool
```

Returns whether an equal KLine exists in the STM tier only.

- Unlike `exists()`, this checks the STM tier only — not Frame, LTM, or Base.
- Uses the same equality semantics as `exists()`: same signature and node
  sequence.

### Iterate STM

```
model.iter_stm() → Iterator[KLine]
```

Returns an iterator over all KLines currently in the STM, in insertion
order (oldest first).

- Returns a fresh iterator on each call.
- Returns an iterator over a **snapshot** taken under the Model's lock (see
  §Thread Safety); it is safe to exhaust after the lock is released.
- External code that needs to iterate STM entries (e.g., for `add_to_ltm()`
  cascades) must use this method.

## Deduplication

All three write methods (`add_to_stm`, `add_to_frame`, `add_to_ltm`) write
unconditionally — there is no early-return guard. Frame and LTM are
monotonic; STM refreshes FIFO position on every write.

## Iteration

### All Klines

```
model.klines() → sequence of Kline
```

Returns all Klines in reverse insertion order (most recent first).

- Includes Klines from all tiers: STM entries first (most recent), then
  Frame entries not in STM, then LTM entries not in Frame, then Base
  entries not in LTM.
- Duplicates across tiers are suppressed: each unique Kline appears once.

### Filtered Iteration

```
model.where(predicate) → sequence of Kline
```

Returns Klines matching a predicate, in reverse insertion order.

- `predicate` — a function `(Kline) → bool`, or an int (KSig) for AND matching.
- Searches all four tiers. Duplicates suppressed.
- The model does not define what predicates are valid. That is caller-defined.

## Graph Traversal

A Kline's nodes are uint64 values. When a node value equals the signature of
another Kline in the model, it forms an **edge** in the model. Graph
traversal resolves these edges.

All graph traversal operations search across all tiers.

### Resolve

```
model.resolve(node) → Kline | none
```

Resolves a node value to the Kline whose signature matches, if one exists.

- Returns `none` if no Kline has that signature in any tier.

### Query Expand

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

### Unpack

```
model.unpack(kline) → sequence of uint64
```

Flattens a kline's signature decomposition into an ordered sequence of
identity signatures (@CONTEXT.md §Identity).

- **Identity** (`is_identity` — empty nodes OR self-referential `{S: [S]}`)
  → `[signature]`. Base case.
- **Canon** (`is_canon` — non-empty, non-self-referential, and
  `signature == make_signature(nodes)`, @signature spec) → the
  concatenation, in node order, of `unpack(child)` for each child kline
  resolved from the node value.
- Any other input (connoted, undersigned, misfit) → raises.
- **Child resolution** uses three-tier precedence (highest first):
  1. empty-nodes identity, 2. genuine canon, 3. self-referential identity.
     The self-referential form is identity but carries no decomposition, so it
     loses to a genuine canon for the same signature — otherwise it would
     displace the canon and collapse the result to identity. Within a kind, the
     most recently added kline wins (@CONTEXT.md §Recency Precedence). If no
     resolvable kline exists for the value → raises.
- Node order is significant (@kline spec): the output sequence preserves
  the node order of every traversed kline.

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

## Thread Safety

The Model is safe for concurrent reads and writes from multiple threads
(e.g. the background Cogitator daemon mutating the model while a subscriber
reads the model from `on_event` on the calling thread).

- **Atomicity.** `Model`, `KLineStore` (Frame/LTM), and the `STM` each hold
  their own `threading.RLock`. Every public mutating and read method acquires
  its lock for the whole operation, so each call is individually atomic. A
  re-entrant lock is required because public methods call other public methods
  on the same object (`unpack` → `_resolve_for_unpack` → `klines` → tier
  chain; `STM.add` → `remove`); a plain `Lock` would self-deadlock on re-entry.
- **Snapshot iterators.** Methods that return iterators (`__iter__`,
  `iter_stm`) materialise a snapshot list **under the lock** and return an
  iterator over that snapshot, so a caller may iterate after the lock is
  released and still observe a consistent point-in-time view. No method
  `yield`s while holding a lock. (`klines()`, `where()`, and `find_all()`
  already return fully-materialised lists, so they are likewise atomic.)
- **Lock ordering.** Strictly **Model → inner tier** (STM / KLineStore / base
  `Model`). The Model always acquires its own lock first, then the inner
  tier's. Inner tiers never call back into a Model, so the ordering is acyclic
  and deadlock-free. Cogitator/agent code therefore needs no locking of its
  own.
- **Base tier.** The optional Base `Model` is read-only after construction, so
  the parent does **not** acquire the Base's lock on reads — unlocked reads are
  safe and this sidesteps lock-ordering discipline.
- **Publish outside the lock.** `KAgent._publish` (which synchronously
  dispatches subscriber callbacks via `adapter.on_event`) is invoked only
  _after_ a Model method has acquired-and-released its own lock. Subscribers
  therefore run **outside** the Model lock and may re-enter the model (on the
  main thread during `rationalise`, on the Cogitator thread during
  `on_s1`/`on_expansion`) without deadlock.

> **What this does and does not guarantee.** The locks guarantee that every
> individual model operation is atomic and that concurrent readers observe
> consistent snapshots (no stale reads, no `RuntimeError` from concurrent
> list mutation). They do **not** guarantee that the rationalisation _event
> count_ is identical regardless of how subscribers inspect events: the
> background Cogitator processes S2/S3 work items asynchronously relative to
> the main-thread `rationalise` loop, and `expand()`'s node-resolution result
> depends on the model state at the moment each work item is processed, so any
> subscriber work (even a bare `time.sleep`) can perturb the interleaving and
> change the count. Count-determinism is owned by the Cogitator/agent layer.

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
  1. It is canonical (`is_canon` — non-empty, non-self-referential, and
     `make_signature(kline.nodes) == kline.signature`), OR
  2. It is countersigned by another kline in the model.
- A self-referential kline `{S: [S]}` is identity, not canon, and is not
  counted as its own countersigner, so it is not S1 by structure.
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
  accumulating a single linear distance, then inverting:
  `(~distance) & MASK64`. Higher significance means closer match.

#### Hyperparameters

- **D_MAX** — maximum distance and maximum significance value
  (`0xFFFF_FFFF_FFFF_FFFF`).
- **MASK64** — 64-bit mask for bitwise inversion (`0xFFFF_FFFF_FFFF_FFFF`).

#### Behavioral Contract

`expand()` must satisfy these properties:

1. **Single distance** — accumulated integer. S3 connotation hops use
   linear distance `S2_S3_DISTANCE + hop_count`. Callers receive significance.
2. **Significance is inverted distance** — `(~distance) & MASK64`.
3. **Distance is topology-driven** — hop distances from graph topology.
4. **S2 signifies short-circuits before S3** — overlap match yields QC and
   stops chain; `s3_connotations` not populated.
5. **Connotation is always S3** — indirect bridging always uses linear distance.
6. **Ungrounded penalty is always +1** — matched but ungrounded nodes.
7. **Bidirectional** — both query and candidate mismatched nodes contribute.
8. **Linear S3 distance** — `S2_S3_DISTANCE + hop_count + _S3_BIAS - 1`;
   `_S3_BIAS = 1`. No quadratic packing.
9. **Recursive expansion** — connotations yielded as `QueryCandidate`.
10. **Cycle detection** — visited signature pairs prevent infinite recursion.
11. **Null-safe resolution** — a `match_sig` yielded by `edge_hops` is
    resolved via `model.find()` rather than asserted to exist. If resolution
    returns None the recursive branch is silently skipped (no exception).
    Unresolvable match signatures are normal graph topology, not errors.

The implementation algorithm and pseudocode are in
`plans/impl/model.md` §2.

#### Per-node contributions

| Node state                                               | Contribution            | Target component       |
| -------------------------------------------------------- | ----------------------- | ---------------------- |
| Mismatched, chain reaches opposing mismatch set at hop N | +N to total_distance    | Accumulated directly   |
| Mismatched, chain reaches signature with bitwise overlap | yields QC, +MAX_HOP     | S2 signifies candidate |
| Mismatched, chain never reaches opposing mismatch set    | +MAX_HOP                | Accumulated directly   |
| Mismatched candidate, chain bridges via connotation      | `S2_S3_DISTANCE + hops` | S3 linear (always)     |
| Matched + structurally grounded (is_s1)                  | 0                       | Neutral                |
| Matched + ungrounded                                     | +1                      | Accumulated directly   |

### Is Countersigned

```
model.is_countersigned(kline) → bool
```

Returns whether a kline is **countersigned** by any kline in the model.

- `kline` — a KLine.
- Returns `true` if the model contains a kline whose signature equals
  `make_signature(kline.nodes)` and whose sole node equals `kline.signature`.
- A self-referential kline `{S: [S]}` is excluded: its nodes_signature is
  `S` and it is itself a one-node kline whose node is `S`, so it would
  otherwise count as its own countersigner.
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

STM eviction removes the oldest Kline from the STM when `add_to_stm()`
would cause the STM to exceed its bound.

- Eviction removes from the STM index only.
- The evicted Kline's signature and nodes signature entries are removed from
  the STM's dual-keyed index.
- Eviction does not affect Frame, LTM, or Base.
- If the evicted Evicted KLines remain discoverable via Frame, LTM, and Base lookups.

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
   `s3_connotations`. Linear distance `S2_S3_DISTANCE + hop_count`.
4. **Unresolved:** Adds `MAX_HOP` (default 100).

Matched-but-ungrounded nodes add 1 each.

### edge_hops Termination

`edge_hops(model, sig)` yields `(hop_count, next_sig)` for each
non-canonical resolution step in the chain
`sig → kline → make_signature(kline.nodes) → ...`. It must terminate on
four conditions without raising:

1. **Cycle** — if the chain revisits a signature already seen in the
   current traversal, it breaks immediately without yielding the cycle.
   Prevents countersigned pairs (e.g. `{M: [H]} ↔ {H: [M]}`) oscillating
   for all MAX_HOP iterations.
2. **Identity kline** — if `make_signature(kline.nodes) == 0` (identity
   kline with empty nodes), it breaks without yielding. There is no
   signature to follow from `nodes = []`.
3. **Canonical kline** — if the kline is canonical
   (`sig == make_signature(nodes)`), it breaks.
4. **Dead end** — if `model.find(sig)` returns None, it breaks.

The total number of yields never exceeds MAX_HOP.

### S3 Bias and Linear Distance

```
_S3_BIAS = 1
S3 connotation distance = S2_S3_DISTANCE + round_trip_hops + _S3_BIAS - 1
```

The bias ensures the minimum S3 distance (one connotation hop) is
`S2_S3_DISTANCE + 1 = 101`, just above the S2|S3 boundary, so S3 distances
always exceed S2 distances while remaining in the same order of magnitude.
Distance grows **linearly** with hop count (settled by the `s3-distance`
auto-tune; the previous quadratic `_pack(d) = d²` was removed).

### Boundaries

Three fixed boundaries classify yielded significance values:

| Boundary | Position           | Meaning                                    |
| -------- | ------------------ | ------------------------------------------ |
| S1\|S2   | `D_MAX`            | Only exact S1 (distance 0) qualifies as S1 |
| S2\|S3   | `~_S2_S3_DISTANCE` | Packed distance threshold (100)            |
| S3\|S4   | `0`                | Only zero-significance is S4               |

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

| ID     | Criterion                                                                       | Origin ref |
| ------ | ------------------------------------------------------------------------------- | ---------- |
| MOD-1  | add_to_frame and find: add KLine via add_to_frame, find by signature returns it | —          |
| MOD-4  | Exists: True after add_to_frame, False before                                   | —          |
| MOD-5  | Find returns most recent KLine when multiple share signature                    | —          |
| MOD-6  | Find_all: returns all KLines with given signature across tiers                  | —          |
| MOD-7  | Find_by_nodes: returns KLine by nodes signature                                 | —          |
| MOD-8  | Remove: removes most recent KLine with given signature                          | —          |
| MOD-9  | Remove never touches base model                                                 | —          |
| MOD-10 | Len returns frame count only (excludes STM and base)                            | —          |

### Four-Tier Lookup

| ID     | Criterion                                               | Origin ref |
| ------ | ------------------------------------------------------- | ---------- |
| MOD-12 | STM priority: KLine in STM found before Frame           | —          |
| MOD-13 | Frame fallback: KLine not in STM found in Frame         | —          |
| MOD-14 | LTM fallback: KLine not in STM/Frame found in LTM       | —          |
| MOD-15 | Base fallback: KLine not in STM/Frame/LTM found in Base | —          |

### Graph Traversal

| ID      | Criterion                                                                                                              | Origin ref |
| ------- | ---------------------------------------------------------------------------------------------------------------------- | ---------- |
| MOD-17  | Resolve: node resolves to KLine via find (node converted to signature form first)                                      | —          |
| MOD-18  | Query_expand depth 0: returns empty                                                                                    | —          |
| MOD-19  | Query_expand depth 2: returns direct children                                                                          | —          |
| MOD-20  | Query_expand cycle detection: no infinite loop                                                                         | —          |
| MOD-22  | Query: find + expand combined                                                                                          | —          |
| MOD-60  | Unpack identity: `unpack({S: []})` → `[S]`                                                                             | —          |
| MOD-61  | Unpack canon: ordered identity child sequence                                                                          | —          |
| MOD-62  | Unpack nested canon: recursively flattened, order preserved                                                            | —          |
| MOD-63  | Unpack non-decomposable input (e.g. connoted) → raises                                                                 | —          |
| MOD-64  | Unpack unresolvable child node → raises                                                                                | —          |
| MOD-65  | Unpack child resolution: empty-identity > canon > self-referential identity                                            | —          |
| MOD-66  | Unpack within-kind ambiguity: most-recently-added wins                                                                 | —          |
| MOD-67  | Unpack self-referential `{S: [S]}` is identity → `[S]` (no recursion); loses to a genuine canon for the same signature | —          |
| MOD-67b | Unpack canon sharing node value but different nodes still recurses                                                     | —          |

### Write Cascade

| ID     | Criterion                                                        | Origin ref |
| ------ | ---------------------------------------------------------------- | ---------- |
| MOD-23 | add_to_stm: always refreshes FIFO (removes-if-present then adds) | —          |
| MOD-24 | add_to_stm: evicts oldest when bound exceeded                    | —          |
| MOD-25 | add_to_frame: writes Frame and cascades to add_to_stm            | —          |
| MOD-26 | add_to_ltm: writes LTM and cascades to add_to_frame              | —          |
| MOD-32 | add_to_frame: Frame is monotonic (append-only)                   | —          |
| MOD-33 | add_to_ltm: LTM is monotonic (append-only)                       | —          |

### Removed Methods

| ID     | Note                                                                   |
| ------ | ---------------------------------------------------------------------- |
| MOD-R1 | `model.add()` removed — replaced by cascade API                        |
| MOD-R2 | `model.promote()` removed — replaced by `add_to_ltm()`                 |
| MOD-R3 | `model.refresh_stm()` removed — absorbed by `add_to_stm()`             |
| MOD-R4 | `model.descendants()` / `model.get_all_descendants()` removed — unused |

### Significance API

| ID     | Criterion                                                                       | Vision ref            |
| ------ | ------------------------------------------------------------------------------- | --------------------- |
| MOD-34 | `is_s1` canonical: genuine canon (`is_canon`) → True                            | @vision §Significance |
| MOD-35 | `is_s1` countersigned: mutual cross-reference → True                            | @vision §Significance |
| MOD-36 | `is_s1` neither: non-canonical, non-countersigned → False                       | —                     |
| MOD-37 | `expand` all-match ungrounded: significance reflects ungrounded count           | —                     |
| MOD-38 | `expand` all-mismatched unresolvable: low significance                          | —                     |
| MOD-39 | `expand` with edge hops: connotation yields + terminal                          | —                     |
| MOD-40 | `expand` S2 signifies: loose match yields QC, terminal still MAX_HOP  | —                     |
| MOD-41 | `expand` S2 before S3: signifies short-circuits connotation recording           | —                     |
| MOD-42 | `expand` S3 route: S3 bias ensures S3 distances exceed S2                       | —                     |
| MOD-43 | `expand` connotation: indirect path → S3 connotation yield + terminal           | —                     |
| MOD-44 | `expand` significance always in valid uint64 range `[1, D_MAX]`                 | —                     |
| MOD-45 | `expand` bidirectional: both sides contribute connotations + terminal           | —                     |
| MOD-46 | `is_countersigned`: mutual node reference detected                              | —                     |
| MOD-47 | Not countersigned: one-way reference → False                                    | —                     |

### Expand & edge_hops Robustness

| ID   | Criterion                                                                  | Origin ref |
| ---- | -------------------------------------------------------------------------- | ---------- |
| ER-1 | Countersigned pair (M↔H) yields at most 2 hops, not MAX_HOP (cycle breaks) | —          |
| ER-2 | Identity kline `{A: []}` yields zero hops                                  | —          |
| ER-6 | `expand()` does not crash when a `match_sig` is unresolvable               | —          |
| ER-7 | S2 expansion scenario over countersign cycles completes without exception  | —          |

### Structural Grounding

| ID      | Criterion                                                                                                                                                                                                 | Origin ref |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| MOD-48  | `promote_participating`: query + candidate promoted after ratification                                                                                                                                    | —          |
| MOD-49  | `promote_participating`: S4 identity klines in STM also promoted                                                                                                                                          | —          |
| MOD-50  | `promote_participating`: S2/S3 partial klines in STM promoted                                                                                                                                             | —          |
| MOD-51  | `promote_participating`: already-promoted klines not re-promoted                                                                                                                                          | —          |
| MOD-52  | `classify_misfit` canonical: `S == N` → (False, False)                                                                                                                                                    | —          |
| MOD-53  | `classify_misfit` underfit: `S & ~N != 0` → (True, False)                                                                                                                                                 | —          |
| MOD-54  | `classify_misfit` overfit: `N & ~S != 0` → (False, True)                                                                                                                                                  | —          |
| MOD-55  | `classify_misfit` dual: both conditions → (True, True)                                                                                                                                                    | —          |
| MOD-56  | `generate_expansions` underfit: returns proposal with added nodes                                                                                                                                         | —          |
| MOD-57  | `generate_expansions` overfit: returns trimmed + companion                                                                                                                                                | —          |
| MOD-58  | `generate_expansions` dual: returns replacement + companion (one atomic swap per gap-filling contributor; the dual path is exclusive — it does not also emit the underfit-only or overfit-only proposals) | —          |
| MOD-59  | `generate_expansions` no gap: no expansion proposals emitted                                                                                                                                              | —          |
| MOD-59b | `generate_expansions` never yields an identity proposal (`{S: []}` or `{S: [S]}`) — identity carries no decomposition; see @cogitator spec §Universal Constraint                                          | —          |

## What a Model is Not

The following are explicitly **out of scope** for this spec:

- **Significance computation.** The model computes significance internally
  via `expand()`. Routing is defined in the significance spec.
- **Encoding.** Converting text or other input into Klines is the agent's
  responsibility (@agent spec).
- **Tokenisation.** Producing nodes from input is defined in the
  @tokenizer spec. Producing signatures from nodes is defined in the
  @signature spec.
- **Persistence format.** The format of the serialised file (sections,
  encoding, versioning) is an implementation-level concern. The spec defines
  that all three mutable tiers (STM, Frame, LTM) are persisted.
- **Debug metadata.** Labels, source text, timestamps, or other diagnostic
  data attached to entries.
- **Thread management.** The choice of synchronisation _mechanism_
  (re-entrant locks, snapshot iterators) is an implementation concern; the
  _contract_ (atomicity, snapshot semantics, lock ordering) is specified in
  §Thread Safety.

## Referenced By

- **Significance** (@significance spec) — calls `is_s1`, `expand`.
- **Agent** (@agent spec) — stores encoded Klines, retrieves candidates,
  traverses the graph, promotes significant Klines.
