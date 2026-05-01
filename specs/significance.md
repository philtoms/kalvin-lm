# Significance Specification

> **Note:** Significance is a *conceptual* specification. The constants (`D_MAX`,
> `MASK64`) and the distance→significance inversion live in `model.py`.
> Routing logic lives in `agent.py`. There is no standalone `significance` module.

## Overview

Significance is a metric that describes how well a candidate Kline answers a
query Kline. It is the bitwise inverse of a packed distance:

```
significance = (~packed_distance) & MASK64
```

The model's `expand()` generator computes this internally and yields
`QueryCandidate` results with significance (not distance). The inversion is a
model-internal concern — callers of `expand()` receive significance directly.

Routing (determining S1/S2/S3/S4 from node membership) is performed by the
Agent's `_route(Q, C)` method.

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

Distance is computed internally by `Model.expand()` as a packed 64-bit value.
The model inverts this to significance before yielding. Callers never see raw
distance — they receive `QueryCandidate.significance` directly.

### Internal packed distance encoding

Distance is a packed 64-bit unsigned integer encoding both S2 and S3
components:

```
  Bits [0, D_PACK_SHIFT-1]     → S2 component
  Bits [D_PACK_SHIFT, 63]      → S3 component
```

`D_PACK_SHIFT` is a model-internal hyperparameter (default: 32).

### Arithmetic Ordering

```
S1 > S2 > S3 > S4
```

Because `~distance` is monotonically decreasing:

```
S1 = 0xFFFF_FFFF_FFFF_FFFF   (all 64 bits set, zero distance)
S4 = 0x0000_0000_0000_0000   (all 64 bits clear, maximum distance)
```

## Model API

The model's `expand()` generator computes significance internally and yields
`QueryCandidate` results:

```
model.expand(Q, C, level) → Iterator[QueryCandidate]
```

- Yields intermediate `QueryCandidate` items for each discovered connotation,
  followed by a terminal `QueryCandidate` with the computed significance.
- `level` is `"S2"` or `"S3"`, determined by routing.
- Each `QueryCandidate.significance` is in range `[1, D_MAX]`.
- The distance→significance inversion is performed inside `expand()`;
  callers use `.significance` directly.

## Properties

1. **Inverted metric**: significance = `(~packed_distance) & MASK64`. Computed
  internally by the model. Higher is more significant.
2. **Routing is self-contained**: routing uses simple node-membership testing
  and does not call any model function.
3. **Pessimistic**: the presence of any unmatched node prevents S1.
4. **Arithmetically comparable**: S1 > S2 > S3 > S4 by unsigned integer
  comparison.
5. **Exhaustive**: every Kline with candidates is S1, S2, or S3.
6. **S1 is trivial**: all nodes match → distance = 0 → significance = D_MAX,
  no function call needed.

## Code Locations

| Concept | File | Symbol |
| ------- | ---- | ------ |
| Constants (`D_MAX`, `MASK64`) | `src/kalvin/model.py` | module-level |
| Routing | `src/kalvin/agent.py` | `Agent._route()` |
| Significance inversion | `src/kalvin/model.py` | `Model.expand()` |
| Distance computation | `src/kalvin/model.py` | `Model.expand()` |
