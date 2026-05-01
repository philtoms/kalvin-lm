# Significance Specification

> **Note:** Significance is a *conceptual* specification. The constants (`D_MAX`,
> `MASK64`) and the distance→significance inversion live in `model.py`.
> Routing logic lives in `agent.py`. There is no standalone `significance` module.

## Overview

Significance is a metric that describes how well a candidate Kline answers a
query Kline. It is the bitwise inverse of a distance:

```
significance = (~distance) & MASK64
```

The model's `expand()` generator computes this internally and yields
`QueryCandidate` results with significance (not distance). The inversion is a
model-internal concern — callers of `expand()` receive significance directly.

Routing (determining S1/S2/S3/S4 from node membership) is performed by the
Agent's `_route(Q, C)` method. Classification against significance boundaries
is performed by the Cogitator's `_classify()` method.

## Constants

Defined in `model.py`:

```python
D_MAX  = 0xFFFF_FFFF_FFFF_FFFF   # maximum distance, also max significance
MASK64 = 0xFFFF_FFFF_FFFF_FFFF   # 64-bit mask for bitwise inversion
```

## Routing

Routing is performed by `Agent._route(Q, C)` — a pure node-membership test
with no model dependency. See @agent spec Phase 5.

```
route(Q, C):
  if Q has no nodes:   return "S4"
  match_count = |{n ∈ Q.nodes : n ∈ C.nodes}|
  if match_count == len(Q.nodes):  return "S1"
  if match_count > 0:              return "S2"
  else:                            return "S3"
```

| Route | Condition | Significance |
| ----- | --------- | ------------ |
| S1    | All query nodes exist in candidate | `D_MAX` (zero distance) |
| S2    | Some query nodes exist in candidate | `model.expand(Q, C, "S2")` → last yield |
| S3    | No query nodes exist in candidate | `model.expand(Q, C, "S3")` → last yield |
| S4    | No candidates or empty query | `0` (maximum distance) |

## Distance

Distance is computed by `Model.expand()` as a single accumulated integer.

### Distance accumulation

For each mismatched node in the query and candidate:

- **Direct edge resolution (S2):** The node resolves via `_edge_hops()` to a
  node in the opposite set. Adds the hop count directly to distance.
- **Connotation resolution (S3):** The node resolves via `_edge_hops()` to a
  signature found in `s3_connotations`. The connotation hop count is biased by
  `1 << _D_PACK_SHIFT` to guarantee S3 distances are astronomically larger
  than S2 distances.
- **Unresolved:** Adds `MAX_HOP` (default 100).

Matched-but-ungrounded nodes (present in both but not resolving to an S1
kline) add 1 each.

### S3 bias

```
s3_hop = connotation_hops
biased_distance = s3_hop << _D_PACK_SHIFT
```

The bias (`1 << 32 ≈ 4.3 billion`) guarantees that any S3 contribution
produces a distance far exceeding any S2 distance (max S2 ≈ `N × MAX_HOP`).
The topology naturally separates the bands — no explicit routing-level
distance assignment needed.

### Significance inversion

```
significance = (~min(distance, D_MAX - 1)) & MASK64
```

The topology guarantees the natural ordering:

| Band | Typical distance range | Typical significance | Condition |
| ---- | ---------------------- | -------------------- | --------- |
| S2   | 0 – ~300               | Near `D_MAX`         | Upper 32 bits of significance all set |
| S3   | `1 << 32` and up       | Mid-range            | Upper 32 bits of significance not all set |
| S4   | `D_MAX`                | 0                    | No candidates |

### Arithmetic Ordering

```
S1 > S2 > S3 > S4
```

Because `~distance` is monotonically decreasing:

```
S1 = 0xFFFF_FFFF_FFFF_FFFF   (all 64 bits set, zero distance)
S4 = 0x0000_0000_0000_0000   (all 64 bits clear, maximum distance)
```

## Significance Boundaries

Classification of yielded significance values is performed against three
boundaries, computed by `Cogitator._boundaries()` and classified by
`Cogitator._classify()`.

### Base boundaries (τ = 1)

| Boundary  | Position                  | Meaning                           |
| --------- | ------------------------- | --------------------------------- |
| S1\|S2    | `D_MAX - 1`               | Only exact S1 qualifies as S1     |
| S2\|S3    | `~(1 << _D_PACK_SHIFT)`   | HP boundary (S3 bias threshold)   |
| S3\|S4    | `0`                       | Only zero-significance is S4      |

### Temperature shift

Temperature shifts all three boundaries in the same direction. Capping
produces the asymmetric effect:

| Direction | S1\|S2       | S2\|S3       | S3\|S4     | Effect                           |
| --------- | ------------ | ------------ | ---------- | -------------------------------- |
| τ > 1     | drops ↓      | drops ↓      | capped at 0 | More S2→S1, more S3→S2          |
| τ < 1     | capped at D_MAX-1 | rises ↑ | rises ↑    | Fewer S2→S1, more S3→S4         |

High temperature lowers the S1|S2 boundary, allowing near-S1 S2
connotations to be classified as S1. Low temperature raises the S3|S4
boundary, demoting weak S3 connotations to S4.

### Classification

```python
def _classify(sig, s12, s23, s34):
    if sig >= s12: return "S1"
    if sig >= s23: return "S2"
    if sig >= s34: return "S3"
    return "S4"
```

## Model API

The model's `expand()` generator computes significance internally and yields
`QueryCandidate` results:

```
model.expand(Q, C, level) → Iterator[QueryCandidate]
```

- Yields intermediate `QueryCandidate` items for each discovered connotation,
  followed by a terminal `QueryCandidate` with the computed significance.
- `level` is `"S2"` or `"S3"`, determined by routing. It is preserved for
  recursive calls but does not affect distance computation.
- Each `QueryCandidate.significance` is in range `[1, D_MAX]`.
- The distance→significance inversion is performed inside `expand()`;
  callers use `.significance` directly.

## Properties

1. **Inverted metric**: significance = `(~distance) & MASK64`. Computed
  internally by the model. Higher is more significant.
2. **Routing is self-contained**: routing uses simple node-membership testing
  and does not call any model function.
3. **Pessimistic**: the presence of any unmatched node prevents S1.
4. **Arithmetically comparable**: S1 > S2 > S3 > S4 by unsigned integer
  comparison.
5. **Exhaustive**: every Kline with candidates is S1, S2, or S3.
6. **S1 is trivial**: all nodes match → distance = 0 → significance = D_MAX,
  no function call needed.
7. **Topology-driven**: distance is accumulated from graph hops. S3
  connotation hops are biased by `1 << _D_PACK_SHIFT`, guaranteeing natural
  S2/S3 separation without routing-level distance assignment.
8. **Boundary classification**: significance values are classified against
  three boundaries (S1|S2, S2|S3, S3|S4) shifted by temperature. Raw
  significance is never mutated per-yield.

## Code Locations

| Concept | File | Symbol |
| ------- | ---- | ------ |
| Constants (`D_MAX`, `MASK64`) | `src/kalvin/model.py` | module-level |
| S3 bias (`_D_PACK_SHIFT`) | `src/kalvin/model.py` | module-level |
| Routing | `src/kalvin/agent.py` | `Agent._route()` |
| Significance inversion | `src/kalvin/model.py` | `Model.expand()` |
| Distance computation | `src/kalvin/model.py` | `Model.expand()` |
| Boundary computation | `src/kalvin/agent.py` | `Cogitator._boundaries()` |
| Classification | `src/kalvin/agent.py` | `Cogitator._classify()` |
| Sampling (top-k, top-p) | `src/kalvin/agent.py` | `Cogitator._run_work_item()` |
| Sampling parameters | `src/kalvin/agent.py` | `Sampling` class |
