# STM (Short-Term Memory) Specification

## Overview

The STM is a **bounded, dual-keyed index** over recently added KLines. It is
the first tier in the Model's three-tier memory architecture (STM → Frame →
Base), providing fast, rolling-window indexing for transitive grounding.

The STM is not a standalone knowledge store — it is an index that the Model
manages internally. KLines added to the STM are always simultaneously added to
the Frame; eviction from the STM removes only the index entry, not the
underlying KLine.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### KLine (@kline spec)

- A KLine is an identified, ordered sequence of zero or more nodes.
- Equality: same signature **and** same node sequence.
- `signature` is a uint64 identity key.

### Signature (@signature spec)

- `make_signature(nodes) → int` — OR-reduction over nodes.
- `is_literal(node) → bool` is defined in the @kline spec (standalone
  bit-layout test, not tokenizer-specific).
- Used to derive the nodes signature for dual-keyed indexing.

## Definition

An STM consists of:

| Component   | Type                                    | Description                             |
| ----------- | --------------------------------------- | --------------------------------------- |
| `_store`    | `dict[KSig, list[KLine]]`              | Signature → bucket of KLines.           |
| `_order`    | `list[KLine]`                           | Insertion order (FIFO for eviction).    |
| `_dedup`    | `set[tuple[KSig, tuple[int, ...]]]`    | Seen (signature, nodes) pairs.          |
| `_bound`    | `int`                                   | Maximum retained KLines (default 256).  |

## Construction

```python
STM(bound=256)
```

- `bound` — maximum number of KLines to retain. Default **256**.
- A newly constructed STM is empty.
- `is_literal` is imported directly from the @kline spec — no injection needed.

## Dual-Keyed Indexing

Each KLine is indexed under **two keys**:

1. **Signature key** — `kline.signature`
2. **Nodes signature key** — `make_signature(kline.nodes)`

Both keys map into the same `_store` dictionary. When the two keys are
identical (signature equals nodes signature), the KLine is stored under a
single key with a single bucket entry.

### Purpose: Transitive Grounding

Nodes-signature indexing enables **transitive grounding**: finding KLines that
share node structure even when their signatures differ. A KLine whose nodes
are structurally similar to another's nodes will share a nodes signature,
allowing the Model to discover grounding relationships that pure signature
lookups would miss.

## API

### Add

```python
stm.add(kline, dedup=True) → bool
```

Adds a KLine to the STM.

- Returns `True` if the KLine was added.
- Returns `False` if rejected as a duplicate (when `dedup=True`).
- When `dedup=True`, checks whether `(kline.signature, tuple(kline.nodes))`
  is already in `_dedup`. If so, returns `False` without modifying state.
- Computes the nodes signature via `make_signature(kline.nodes)`. If the
  KLine has no nodes, the nodes signature is `0`.
- The effective signature key is `kline.signature` if non-zero, otherwise
  the computed nodes signature.
- **Enforces bound**: if `len(_order) >= bound` before adding, evicts the
  oldest entry (see Eviction).
- Indexes the KLine under both the signature key and the nodes signature key
  (if different and non-zero).

### Get

```python
stm.get(key) → list[KLine]
```

Returns all KLines indexed under `key`, or an empty list.

- Returns a **copy** of the bucket (safe for callers to mutate).
- O(1) dictionary lookup.

### Get KLine

```python
stm.get_kline(key) → KLine | None
```

Returns the **most recently added** KLine under `key`, or `None`.

- Returns the last element of the bucket for `key`.

### Find by Signature

```python
stm.find_by_signature(signature) → list[KLine]
```

Equivalent to `stm.get(signature)`.

### Find by Nodes

```python
stm.find_by_nodes(nodes_signature) → list[KLine]
```

Equivalent to `stm.get(nodes_signature)`.

### Query

```python
stm.query(sig) → list[KLine]
```

Returns all KLines whose signatures share at least one bit with `sig`
(bitwise AND ≠ 0).

- Returns an empty list if `sig == 0` (unsigned signatures cannot match).
- Results are in **reverse insertion order** (most recent first).
- Each KLine appears at most once (deduped by identity).

### Remove

```python
stm.remove(kline) → None
```

Removes a KLine from all index entries.

- Removes from the signature key bucket.
- Removes from the nodes signature key bucket (if different and non-zero).
- Removes from `_order`.
- Removes from `_dedup`.
- If a bucket becomes empty after removal, the key is deleted from `_store`.
- No error if the KLine is not in the STM (silent no-op).

### Clear

```python
stm.clear() → None
```

Removes all entries. Resets to empty state.

### Length

```python
len(stm) → int
```

The number of KLines currently in the STM. Equal to `len(_order)`.

### Contains

```python
key in stm → bool
```

Returns whether `key` has any indexed KLines. Checks `_store` directly.

## Eviction

Eviction removes the **oldest** KLine when `add` would cause the STM to
exceed its bound.

### FIFO Order

The `_order` list maintains insertion order. The oldest KLine is the first
element. On eviction:

1. Pop the first element from `_order`.
2. Remove the KLine from its signature key bucket in `_store`.
3. Remove the KLine from its nodes signature key bucket (if different and
   non-zero).
4. Remove the `(signature, nodes)` pair from `_dedup`.
5. If a bucket becomes empty, delete the key from `_store`.

### Properties

- Eviction removes from the STM index **only**. The KLine remains in the
  Frame and Base (managed by the Model, not the STM).
- Evicted KLines are still discoverable via Frame and Base lookups.
- The STM bound is enforced strictly: `len(stm) <= bound` is an invariant
  after every `add` call.

## Deduplication

When `dedup=True` (the default), `add` rejects KLines that duplicate an
existing entry. A duplicate is defined as a KLine with the same
`(signature, tuple(nodes))` pair already in `_dedup`.

- Deduplication is **within the STM only** — it does not check the Frame
  or Base (that is the Model's responsibility).
- The `_dedup` set is maintained in sync with `_order`: eviction and
  removal both discard the corresponding dedup entry.

## Properties

1. **Bounded** — `len(stm) <= bound` is always true after any operation.
2. **Dual-keyed** — every KLine is retrievable by both signature and nodes
   signature.
3. **FIFO eviction** — oldest entries are evicted first; no priority or
   significance-based selection.
4. **Identity-based buckets** — multiple KLines may share a key; they are
   stored as an ordered list within the bucket.
5. **Copy-on-read** — `get` returns a copy of the bucket; callers cannot
   corrupt internal state.
6. **No external side effects** — STM operations do not affect the Frame
   or Base. The Model manages cross-tier consistency.

## What an STM is Not

The following are explicitly **out of scope** for this spec:

- **Persistence.** STM state is not serialised; it is rebuilt from the
  Frame on session start.
- **Significance computation.** The STM provides indexing; significance is
   computed by the Model and Agent.
- **Cross-tier consistency.** The Model owns the STM/Frame/Base
  relationship. The STM does not know about other tiers.
- **Thread safety.** Concurrent access management is an implementation
   concern.
- **Nodes signature computation.** The STM delegates to `make_signature`
  from the @signature spec, which uses `is_literal` from the @kline spec.

## Referenced By

- **Model** (@model spec) — owns and manages the STM as its first tier.
- **Agent** (@agent spec) — indirectly, via Model operations that populate
  the STM.

## Code Location

| Symbol | File              |
| ------ | ----------------- |
| `STM`  | `src/kalvin/stm.py` |
