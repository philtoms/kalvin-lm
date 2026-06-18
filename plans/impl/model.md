# Sub-Plan: Model — Four-Tier Memory + Distance Algorithm

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Phase:** 5
**Estimate:** 2–3 days
**Depends on:** Foundations (KLine, Signature, STM — see @stm spec)

---

## 1. Spec References

See **@model spec** for full definition (construction, storage operations, graph
traversal, promotion, STM refresh, significance API, deduplication rules).
Test matrix: MOD-1 through MOD-55.

See **@stm spec** for STM definition.

### Key API (from spec)

```python
model.add_to_stm(kline) → None       # STM only (always refreshes FIFO)
model.add_to_frame(kline) → None     # Frame + cascades to add_to_stm()
model.add_to_ltm(kline) → None       # LTM + cascades to add_to_frame()
model.exists(kline) → bool       # Check across all tiers
model.find(signature) → KLine|None
model.find_all(signature) → list[KLine]
model.find_by_nodes(nodes_sig) → KLine|None
len(model) → int                 # Frame count only
model.klines() → list[KLine]
model.where(predicate) → list[KLine]
model.resolve(node) → KLine|None
model.query_expand(kline, depth=2) → list[KLine]
model.descendants(node) → set[int]
model.query(signature, depth=1) → list[KLine]
model.expand(query, candidate) → Iterator[QueryCandidate]
model.is_countersigned(kline) → bool
model.is_s1(kline) → bool
model.classify_misfit(kline) → tuple[bool, bool]
model.generate_expansions(kline, underfit_gap, overfit_mask) → Iterator
```

---

## 2. Expand Algorithm

### `expand(query, candidate, distance=0) → Iterator[QueryCandidate]`

A generator that expands a query-candidate pair, yielding intermediate
connotation results and a terminal packed distance. Replaces the previous
`distance()` function with a richer expansion API.

- `distance` — accumulated hop distance for recursive calls (default 0).
- **Yields** intermediate `QueryCandidate` items for connotations, then a
  terminal `QueryCandidate` with the computed significance.
- **Recursive**: uses `yield from expand(...)` for connotation expansion.
- **Cycle detection**: `_visited` set of `(query.sig, candidate.sig)` pairs.

### Hyperparameters

- **MAX_HOP** — upper bound on edge hop chain depth (default 100). Also the
  penalty for unresolvable mismatched nodes.
- **\_S3_BIAS** — tier bias for S3 connotation hops (default 1). Connotation
  distance is `S2_S3_DISTANCE + hops + _S3_BIAS - 1`; the bias seats the
  minimum S3 distance one above the S2|S3 boundary.

### Definitions

- **is_s1(kline)** — kline is structurally grounded: canonical
  (`make_signature(nodes) == signature`) or countersigned.
- **is_canon(kline)** — `kline.signature == make_signature(kline.nodes)`.
  Canonical klines are terminals in the resolution chain.
- **edge_hops(sig)** — yields `(hop_count, next_sig)` for each non-canonical
  resolution step.

```python
def _edge_hops(self, sig: int) -> Iterator[tuple[int, int]]:
    hop_count = 0
    while hop_count < MAX_HOP:
        kline = self.find(sig)
        if kline is None or self._is_canon(kline):
            break
        hop_count += 1
        sig = make_signature(kline.nodes)
        yield hop_count, sig
```

### Algorithm

```python
def expand(self, query, candidate, distance=0, _visited=None):
    if _visited is None:
        _visited = set()
    key = (query.signature, candidate.signature)
    if key in _visited:
        return
    _visited.add(key)

    q_set = set(query.nodes)
    c_set = set(candidate.nodes)
    mismatched_q = q_set - c_set
    mismatched_c = c_set - q_set
    matched = q_set & c_set

    total_distance = distance
    s3_connotations = {}  # sig → min hops from any query node

    # Pass 1: mismatched query nodes
    #   - Direct match in mismatched_c → yield S2 connotation (exact)
    #   - Signifies match → yield S2 cogitation candidate (loose)
    #   - No match → record in s3_connotations for bridging
    for n in mismatched_q:
        hop_distance = MAX_HOP
        for hops, match_sig in self._edge_hops(n):
            if match_sig in mismatched_c:
                hop_distance = hops
                q_kline = self.find(n)
                c_kline = self.find(match_sig)
                if q_kline and c_kline:
                    yield from self.expand(q_kline, c_kline, hops)
                break
            elif signifies(n, match_sig):
                c_kline = self.find(match_sig)
                if c_kline is not None:
                    sig_distance = distance + hops
                    significance = (~min(sig_distance, D_MAX - 1)) & MASK64
                    yield QueryCandidate(self.find(n), c_kline, significance)
                break
            elif (match_sig not in s3_connotations
                  or hops < s3_connotations[match_sig]):
                s3_connotations[match_sig] = hops
        total_distance += hop_distance

    # Pass 2: mismatched candidate nodes
    #   - Direct match in mismatched_q → yield S2 connotation (exact)
    #   - Signifies match → yield S2 cogitation candidate (loose)
    #   - Connotation bridge → yield S3 connotation
    for n in mismatched_c:
        hop_distance = MAX_HOP
        for hops, match_sig in self._edge_hops(n):
            if match_sig in mismatched_q:
                hop_distance = hops
                q_kline = self.find(n)
                c_kline = self.find(match_sig)
                if q_kline and c_kline:
                    yield from self.expand(q_kline, c_kline, hops)
                break
            elif signifies(n, match_sig):
                c_kline = self.find(match_sig)
                if c_kline is not None:
                    sig_distance = distance + hops
                    significance = (~min(sig_distance, D_MAX - 1)) & MASK64
                    yield QueryCandidate(self.find(n), c_kline, significance)
                break
            elif match_sig in s3_connotations:
                hop_distance += s3_connotations[match_sig] + hops
                q_kline = self.find(n)
                c_kline = self.find(match_sig)
                if q_kline and c_kline:
                    yield from self.expand(q_kline, c_kline, hop_distance)
                hop_distance = 0
                break
        total_distance += hop_distance

    # Matched but not grounded → penalty
    for n in matched:
        kl = self.find(n)
        if kl is None or not self.is_s1(kl):
            total_distance += 1

    # Carry forward the incoming distance
    total_distance += distance

    # Clamp to avoid overflow, then invert to significance
    significance = (~min(total_distance, D_MAX - 1)) & MASK64
    yield QueryCandidate(query, candidate, significance)
```

### Per-Node Contributions

| Node state                                               | Contribution         | Target component       |
| -------------------------------------------------------- | -------------------- | ---------------------- |
| Mismatched, chain reaches opposing mismatch set at hop N | +N to total_distance | Accumulated directly   |
| Mismatched, chain reaches signature with bitwise overlap | yields QC, +MAX_HOP  | S2 signifies candidate |
| Mismatched, chain never reaches opposing mismatch set    | +MAX_HOP             | Accumulated directly   |
| Mismatched candidate, chain bridges via connotation      | round-trip, hop = 0  | S3 packed (always)     |
| Matched + grounded (is_s1)                                | 0                     | Neutral               |
| Matched + ungrounded                                     | +1                   | Accumulated directly   |

### Connotation Bridging

When no direct hop path exists between mismatched query and candidate nodes,
the algorithm attempts **connotation bridging**: query nodes that reach
intermediate signatures during their hop chains record the minimum hop count
to each intermediate. Candidate nodes then check whether their hop chains
reach any of these intermediates. If so, the round-trip distance
(query hops + candidate hops) contributes to the S3 component.

Connotation bridging always contributes to S3 regardless of routing.
This captures indirect, associative connections — the "reminiscent"
relationship that characterises S3.

### S2 Signifies Cogitation

Before connotation bridging is attempted, each hop in the chain is checked
for **signifies** — whether the node value and the reached signature share
at least one set bit (`signifies(node, reached_sig)`). If so, a
`QueryCandidate` is yielded with `significance = (~min(distance + hops, D_MAX - 1)) & MASK64`.

This captures a looser structural relationship than exact matching: the
signatures are not identical, but they overlap in at least one bit,
suggesting potential significance. The candidate is yielded for cogitation
without further recursive expansion. The mismatched node still contributes
`MAX_HOP` to the terminal distance — signifies does not resolve the mismatch.

Signifies short-circuits before S3 connotation recording, preventing the
same hop from being recorded as both an S2 and an S3 path.

### Connotation Expansion

When `expand()` discovers a direct hop between a mismatched query node and a
mismatched candidate node, it **yields an S2 connotation**: a recursive
`expand()` call on the resolved KLines. This represents a direct structural
relationship between sub-graphs.

When connotation bridging succeeds, `expand()` **yields an S3 connotation**:
a recursive call with linear distance
`S2_S3_DISTANCE + round_trip_hops + _S3_BIAS - 1` (`_S3_BIAS = 1`). This
captures indirect, associative connections; distance grows linearly with hop
count (the previous quadratic `_pack(d) = d²` was removed by the
`s3-distance` auto-tune).

Each connotation is a `QueryCandidate` processed identically by the
Cogitator — countersignature is checked for every yielded item.

### Properties

1. **Single distance** — accumulated integer. S3 connotation hops use
   linear distance `S2_S3_DISTANCE + hops`. Callers receive significance,
   not distance.
2. **Significance is inverted distance** — the model inverts distance
   to produce significance: `(~distance) & MASK64`.
3. **Level is preserved but distance is topology-driven** — mismatched hop
   distances accumulate purely from topology.
4. **S2 signifies short-circuits before S3** — when a node's edge hop
   reaches a signature with bitwise overlap, a `QueryCandidate` is yielded
   for cogitation and the hop chain stops. The `s3_connotations` dict is
   not populated for that hop.
5. **Connotation is always S3** — indirect bridging always uses linear distance.
6. **Ungrounded penalty is always +1** — matched but ungrounded nodes add 1.
7. **Bidirectional** — mismatched nodes from both query and candidate
   contribute.
8. **Linear S3 distance** — `S2_S3_DISTANCE + hops + _S3_BIAS - 1`
   (`_S3_BIAS = 1`); small and large distances stay proportionally close.

### `is_countersigned(a, b) → bool`

```python
def is_countersigned(self, a: KLine, b: KLine) -> bool:
    return (b.signature in a.nodes) and (a.signature in b.nodes)
```

---

## 3. Test Mapping

### Storage

| Spec ID | Test                      | Description                      |
| ------- | ------------------------- | -------------------------------- |
| MOD-1   | add_to_frame and find         | Add KLine via add_to_frame, find returns it |
| MOD-4   | Exists                    | True after add_to_frame, False before |
| MOD-5   | Find returns most recent  | Multiple KLines same sig         |
| MOD-6   | Find_all                  | Returns all KLines with sig      |
| MOD-7   | Find_by_nodes             | Returns by nodes signature       |
| MOD-8   | Remove                    | Removes most recent with sig     |
| MOD-9   | Remove never touches base | Verify base unchanged            |
| MOD-10  | Len                       | Frame count only                 |

### Four-Tier Lookup

| Spec ID | Test             | Description                                  |
| ------- | ---------------- | -------------------------------------------- |
| MOD-12  | STM priority     | KLine in STM found before Frame              |
| MOD-13  | Frame fallback   | KLine not in STM found in Frame              |
| MOD-14  | LTM fallback     | KLine not in STM/Frame found in LTM          |
| MOD-15  | Base fallback    | KLine not in STM/Frame/LTM found in Base     |

### Graph Traversal

| Spec ID | Test                   | Description             |
| ------- | ---------------------- | ----------------------- |
| MOD-17  | Resolve                | Node resolves to KLine  |
| MOD-18  | Expand depth 0         | Returns empty           |
| MOD-19  | Expand depth 2         | Returns direct children |
| MOD-20  | Expand cycle detection | No infinite loop        |
| MOD-21  | Descendants            | Recursive collection    |
| MOD-22  | Query                  | Find + expand           |

### Write Cascade

| Spec ID | Test                | Description                                         |
| ------- | ------------------- | --------------------------------------------------- |
| MOD-23  | add_to_stm refresh     | Removes-if-present then adds, refreshing FIFO       |
| MOD-24  | add_to_stm evict       | Oldest evicted when bound exceeded                  |
| MOD-25  | add_to_frame cascade   | Writes Frame and cascades to add_to_stm                |
| MOD-26  | add_to_ltm cascade     | Writes LTM and cascades to add_to_frame                |
| MOD-32  | add_to_frame monotonic | Frame is append-only                                |
| MOD-33  | add_to_ltm monotonic   | LTM is append-only                                  |

### Significance API

| Spec ID | Test                      | Description                                                |
| ------- | ------------------------- | ---------------------------------------------------------- |
| MOD-34  | is_s1 canonical            | `make_signature(nodes) == signature` → True                |
| MOD-35  | is_s1 countersigned        | Mutual cross-reference detected → True                     |
| MOD-36  | is_s1 neither              | Non-canonical, non-countersigned → False                   |
| MOD-37  | expand self no model      | All match, ungrounded → significance reflects N ungrounded |
| MOD-38  | expand no resolution      | All mismatched unresolvable → low significance             |
| MOD-39  | expand grounding          | Matched node that resolves → higher significance           |
| MOD-40  | expand edge hops          | Mismatched with chain → connotation yields + terminal      |
| MOD-41  | expand S2 signifies       | Signifies loose match yields QC, terminal still MAX_HOP    |
| MOD-42  | expand S2 before S3       | Signifies short-circuits, s3_connotations not populated    |
| MOD-43  | expand range              | Valid significance uint64                                  |
| MOD-44  | expand clamped            | Significance in [1, D_MAX]                                 |
| MOD-45  | expand S3 route           | S3 bias ensures S3 distances moderately exceed S2          |
| MOD-46  | expand connotation        | Indirect path → S3 connotation yield + terminal            |
| MOD-47  | expand significance range | Significance always in valid uint64 range                  |
| MOD-48  | expand bidirectional      | Both sides yield connotations + terminal                   |
| MOD-49  | is_countersigned          | Mutual reference detected                                  |
| MOD-50  | Not countersigned         | One-way reference → False                                  |
