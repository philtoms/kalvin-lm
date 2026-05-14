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
| S2    | Some query nodes exist in candidate | `model.expand(Q, C)` → last yield |
| S3    | No query nodes exist in candidate | `model.expand(Q, C)` → last yield |
| S4    | No candidates or empty query | `0` (maximum distance) |

## Distance

Distance is computed by `Model.expand()` as a single accumulated integer.

### Distance accumulation

For each mismatched node in the query and candidate, the hop chain is
traversed with a three-tier priority:

1. **Exact match (S2 direct):** The node resolves via `_edge_hops()` to a
   node in the opposite mismatch set. Adds the hop count directly to
   distance and recursively yields an S2 connotation.
2. **Signifies match (S2 loose):** The node resolves via `_edge_hops()` to a
   signature that shares bits with the node value (`signifies(n, sig) ==
   true`). Yields a `QueryCandidate` for cogitation with significance
   `(~min(distance + hops, D_MAX - 1)) & MASK64`. The node still contributes
   `MAX_HOP` to terminal distance. Short-circuits before S3 connotation
   recording.
3. **Connotation resolution (S3):** The node resolves via `_edge_hops()` to
   a signature found in `s3_connotations`. The connotation hop count is
   packed via `_pack(hop_count + _S3_BIAS)` to ensure S3 distances
   moderately exceed S2 distances while remaining close enough for
   potential bridging via further graph expansion.
4. **Unresolved:** Adds `MAX_HOP` (default 100).

Matched-but-ungrounded nodes (present in both but not resolving to an S1
kline) add 1 each.

### S3 bias

```
s3_hop = connotation_hops
biased_distance = _pack(s3_hop + _S3_BIAS)
```

The bias (`_S3_BIAS = 9`, minimum packed distance = `_pack(2+9) = 121`)
ensures S3 distances moderately exceed S2 distances while keeping both tiers
in the same order of magnitude.
The quadratic `_pack` function (d²) compresses small distances together and
spreads large distances apart, ensuring accumulation penalties grow
super-linearly.

### Significance inversion

```
significance = (~min(distance, D_MAX - 1)) & MASK64
```

The topology guarantees the natural ordering:

| Band | Typical distance range | Typical significance | Condition |
| ---- | ---------------------- | -------------------- | --------- |
| S2   | 0 – ~100               | Near `D_MAX`         | Above S2|S3 boundary |
| S3   | `_pack(2+9)=121` and up | Mid-range            | Below S2|S3 boundary |
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

### Base boundaries

| Boundary  | Position                  | Meaning                           |
| --------- | ------------------------- | --------------------------------- |
| S1\|S2    | `D_MAX - 1`               | Only exact S1 qualifies as S1     |
| S2\|S3    | `~_S2_S3_DISTANCE`        | Packed distance threshold (100)   |
| S3\|S4    | `0`                       | Only zero-significance is S4      |

### Fixed Boundaries

The three boundaries are fixed by the distance algorithm and are not
parameterized. Proposals can be emitted at any significance level —
Kalvin does not suppress results below a threshold.

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
model.expand(Q, C) → Iterator[QueryCandidate]
```

- Yields intermediate `QueryCandidate` items for each discovered connotation,
  followed by a terminal `QueryCandidate` with the computed significance.
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
  connotation hops are biased by `_pack(hop_count + _S3_BIAS)`, ensuring
  S3 distances moderately exceed S2 while keeping both tiers in the same
  order of magnitude.
8. **Boundary classification**: significance values are classified against
  three fixed boundaries (S1|S2, S2|S3, S3|S4). Raw significance is never
  mutated per-yield.
9. **S2 expansion**: when countersignature fails for an S2 result, the
  Cogitator may attempt to reshape the candidate kline's nodes to better
  match its signature. See `docs/extended-cogitation.md`.

## Code Locations

| Concept | File | Symbol |
| ------- | ---- | ------ |
| Constants (`D_MAX`, `MASK64`) | `src/kalvin/model.py` | module-level |
| S3 bias (`_S3_BIAS`) | `src/kalvin/model.py` | module-level |
| Distance packing (`_pack`) | `src/kalvin/model.py` | module-level |
| S2|S3 threshold (`_S2_S3_DISTANCE`) | `src/kalvin/agent.py` | module-level |
| Routing | `src/kalvin/agent.py` | `Agent._route()` |
| Significance inversion | `src/kalvin/model.py` | `Model.expand()` |
| Distance computation | `src/kalvin/model.py` | `Model.expand()` |
