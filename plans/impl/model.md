# Sub-Plan: Model — Three-Tier Knowledge Graph + Distance Algorithm

**Parent:** [`plans/implement-kalvin.md`](../implement-kalvin.md)
**Phase:** 5
**Estimate:** 2–3 days
**Depends on:** Foundations (KLine, Signature, STM — see @stm spec)

---

## 1. Model Spec

Three-tier layered knowledge graph: `STM → Frame → Base`. STM is defined in the **@stm spec**.

### Tier Roles

| Tier  | Purpose               | Bounded   | Lifetime       | Modified by `add`   |
| ----- | --------------------- | --------- | -------------- | ------------------- |
| STM   | Transitive grounding  | Yes (256) | Rolling window | Yes                 |
| Frame | Session write surface | No        | Per-session    | Yes                 |
| Base  | Long-term knowledge   | No        | Persistent     | No (promotion only) |

**Lookup order:** STM → Frame → Base. Callers see a single unified API.

### Construction

```python
Model(base=None, stm_bound=256)
```

### Storage Operations

```python
model.add(kline) → bool           # Add to STM + Frame
model.exists(kline) → bool       # Check across all tiers
model.find(signature) → KLine|None       # Most recent by sig
model.find_all(signature) → list[KLine]  # All by sig
model.find_by_nodes(nodes_sig) → KLine|None  # By nodes signature
model.remove(signature) → bool   # Remove most recent, never from base
len(model) → int                 # Frame count only
```

### Deduplication

When `add` receives a literal KLine (all nodes literal per the standalone
`is_literal` function),
check for an equal KLine in any tier. If duplicate exists, reject
(return False). Non-literal KLines are always accepted.

### Iteration

```python
model.klines() → list[KLine]     # All KLines, reverse insertion order, deduped
model.where(predicate) → list[KLine]  # Filtered; if predicate is int, AND match
```

### Graph Traversal

```python
model.resolve(node) → KLine|None     # node → KLine lookup (= find)
model.query_expand(kline, depth=2) → list[KLine]  # Graph expansion
model.descendants(node) → set[int]   # Recursive node collection
model.query(signature, depth=1) → list[KLine]  # Find + expand
```

**Query_expand semantics:**

- `depth=0` → `[]`
- `depth=1` → `[]`
- `depth=2` → direct children of kline
- `depth=N` → N-1 levels deep
- Cycle detection: visited KLines not re-visited.
- Missing nodes skipped.
- Order: node order, depth-first.

### Promotion

```python
model.promote(kline) → bool    # Add to base. Returns False if no base.
model.promote_all() → int      # Promote all frame KLines to base.
```

### Significance API (consumed by Cogitator)

```python
model.expand(query, candidate) → Iterator[QueryCandidate]  # connotation generator
model.is_countersigned(a, b) → bool
```

Routing is performed by the Agent's `_route()` method using node-membership
testing — no model function is called during routing. The Cogitator
consumes `expand` and `is_countersigned`.

`expand()` is a generator that yields intermediate `QueryCandidate` items
for each discovered connotation, followed by a terminal yield with the
packed distance. Each yielded result is processed by the Cogitator
(checking countersignature), enabling discovery across the full expansion
graph.

```python
class QueryCandidate(NamedTuple):
    query: KLine
    candidate: KLine
    significance: int   # pre-computed by model: (~packed_distance) & MASK64
```

### Model-Internal Functions

```python
model.is_s1(kline) → bool              # structural grounding check
model._is_canon(kline) → bool          # sig == make_signature(nodes)
model._edge_hops(sig) → Iterator       # yields (hop_count, next_sig)
```

- `is_s1`: `True` when a kline is structurally grounded — canonical
  (`make_signature(nodes) == signature`) or countersigned by another kline.
- `_is_canon`: tests whether a kline is canonical (its signature equals the
  `make_signature` reduction of its nodes).
- `_edge_hops`: yields `(hop_count, next_sig)` pairs for each non-canonical
  resolution step. The `distance` algorithm consumes it.

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
- **\_S3_BIAS** — tier bias for S3 connotation hops (default 9). Connotation
  hop counts are biased by this amount before quadratic packing.
- **\_pack(distance)** — quadratic packing function: `d²`. Compresses small
  distances together and spreads large distances apart.

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
a recursive call with `_pack(round_trip_hops + _S3_BIAS)`. This captures
indirect, associative connections with quadratic packing.

Each connotation is a `QueryCandidate` processed identically by the
Cogitator — countersignature is checked for every yielded item.

### Properties

1. **Single distance** — accumulated integer. S3 connotation hops biased
   by `_pack(hops + _S3_BIAS)`. Callers receive significance, not distance.
2. **Significance is inverted distance** — the model inverts distance
   to produce significance: `(~distance) & MASK64`.
3. **Level is preserved but distance is topology-driven** — mismatched hop
   distances accumulate purely from topology.
4. **S2 signifies short-circuits before S3** — when a node's edge hop
   reaches a signature with bitwise overlap, a `QueryCandidate` is yielded
   for cogitation and the hop chain stops. The `s3_connotations` dict is
   not populated for that hop.
5. **Connotation is always S3** — indirect bridging always uses packed distance.
6. **Ungrounded penalty is always +1** — matched but ungrounded nodes add 1.
7. **Bidirectional** — mismatched nodes from both query and candidate
   contribute.
8. **Quadratic packing** — `_pack(d) = d²` ensures small distances stay
   close and large distances spread apart.

### `is_countersigned(a, b) → bool`

```python
def is_countersigned(self, a: KLine, b: KLine) -> bool:
    return (b.signature in a.nodes) and (a.signature in b.nodes)
```

Structural test only. Literal nodes cannot match a signature.

---

## 3. Test Cases

### Storage

| Test                      | Description                      |
| ------------------------- | -------------------------------- |
| Add and find              | Add KLine, find by signature     |
| Add returns True          | Normal case                      |
| Literal dedup             | Duplicate literal KLine rejected |
| Non-literal no dedup      | Duplicate non-literal accepted   |
| Exists                    | True after add, False before     |
| Find returns most recent  | Multiple KLines same sig         |
| Find_all                  | Returns all KLines with sig      |
| Find_by_nodes             | Returns by nodes signature       |
| Remove                    | Removes most recent with sig     |
| Remove never touches base | Verify base unchanged            |
| Len                       | Frame count only                 |

### Three-Tier Lookup

| Test             | Description                            |
| ---------------- | -------------------------------------- |
| STM priority     | KLine in STM found before frame        |
| Frame fallback   | KLine not in STM found in frame        |
| Base fallback    | KLine not in STM/frame found in base   |
| Cross-tier dedup | Literal KLine in base blocks frame add |

### Graph Traversal

| Test                   | Description             |
| ---------------------- | ----------------------- |
| Resolve                | Node resolves to KLine  |
| Expand depth 0         | Returns empty           |
| Expand depth 2         | Returns direct children |
| Expand cycle detection | No infinite loop        |
| Descendants            | Recursive collection    |
| Query                  | Find + expand           |

### Promotion

| Test                     | Description                          |
| ------------------------ | ------------------------------------ |
| Promote to base          | KLine appears in base                |
| Promote without base     | Returns False                        |
| Promote all              | All frame KLines promoted            |
| Promote skips base dupes | Existing base KLines not overwritten |

### Significance API

| Test                      | Description                                                |
| ------------------------- | ---------------------------------------------------------- |
| is_s1 canonical            | `make_signature(nodes) == signature` → True                |
| is_s1 countersigned        | Mutual cross-reference detected → True                     |
| is_s1 neither              | Non-canonical, non-countersigned → False                   |
| \_is_canon match          | `sig == make_signature(nodes)` → True                      |
| \_is_canon mismatch       | `sig != make_signature(nodes)` → False                     |
| \_edge_hops unresolvable  | Node doesn't resolve → empty generator                     |
| \_edge_hops canonical     | Resolves to canonical → empty generator                    |
| \_edge_hops chain         | Yields (hop_count, next_sig) at each step                  |
| expand self no model      | All match, ungrounded → significance reflects N ungrounded |
| expand no resolution      | All mismatched unresolvable → low significance             |
| expand grounding          | Matched node that resolves → higher significance           |
| expand edge hops          | Mismatched with chain → connotation yields + terminal      |
| expand S2 signifies       | Signifies loose match yields QC, terminal still MAX_HOP    |
| expand S2 before S3       | Signifies short-circuits, s3_connotations not populated    |
| expand range              | Valid significance uint64                                  |
| expand clamped            | Significance in [1, D_MAX]                                 |
| expand S3 route           | S3 bias ensures S3 distances moderately exceed S2          |
| expand connotation        | Indirect path → S3 connotation yield + terminal            |
| expand significance range | Significance always in valid uint64 range                  |
| expand bidirectional      | Both sides yield connotations + terminal                   |
| is_countersigned          | Mutual reference detected                                  |
| Not countersigned         | One-way reference → False                                  |
