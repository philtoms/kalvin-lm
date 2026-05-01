# Significance Specification

> **Note:** Significance is a *conceptual* specification. The constants (`D_MAX`,
> `MASK64`) and routing logic live in `agent.py`. Graph expansion (including
> distance computation) lives in `model.py`. There is no standalone
> `significance` module.

## Overview

Significance is a metric that describes how well a candidate Kline answers a
query Kline. It is computed as the inverse of a distance in a metric space:

```
significance = ~distance
```

Routing (determining S1/S2/S3/S4 from node membership) is performed by the
Agent's `_route(Q, C)` method. The constants and arithmetic definitions used
by the Agent and the Model are defined in `agent.py`.

## Constants

Defined in `agent.py`:

```python
D_MAX  = 0xFFFF_FFFF_FFFF_FFFF   # maximum distance (S4)
MASK64 = 0xFFFF_FFFF_FFFF_FFFF   # 64-bit mask for inversion
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

| Route | Condition | Distance |
| ----- | --------- | -------- |
| S1    | All query nodes exist in candidate | 0 |
| S2    | Some query nodes exist in candidate | `model.expand(Q, C, "S2")` → last yield |
| S3    | No query nodes exist in candidate | `model.expand(Q, C, "S3")` → last yield |
| S4    | No candidates or empty query | `D_MAX` |

## Distance

Distance is computed by `Model.expand(Q, C, level)`, a generator that yields
intermediate connotation results followed by a terminal `QueryCandidate`
with the packed distance. Semantics are defined in the @model spec.
The significance layer performs the inversion on each yielded result:

```
for qc in model.expand(Q, C, level):
    significance = (~qc.distance) & MASK64
    # process qc (check countersignature, etc.)
```

### Packed Distance Encoding

Distance is a packed 64-bit unsigned integer encoding both S2 and S3
components:

```
  Bits [0, D_PACK_SHIFT-1]     → S2 component
  Bits [D_PACK_SHIFT, 63]      → S3 component
```

`D_PACK_SHIFT` is a hyperparameter (default: 32).

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

The significance layer consumes the following model function:

```
model.expand(Q, C, level) → Iterator[QueryCandidate]
```

- Yields intermediate `QueryCandidate` items for each discovered connotation,
  followed by a terminal `QueryCandidate` with the packed distance.
- `level` is `"S2"` or `"S3"`, determined by routing.
- Each `QueryCandidate` has `.distance` clamped to `[0, D_MAX - 1]`.

## Properties

1. **Inverted metric**: significance = ~distance. Higher is more significant.
2. **Routing is self-contained**: routing uses simple node-membership testing
  and does not call any model function.
3. **Pessimistic**: the presence of any unmatched node prevents S1.
4. **Arithmetically comparable**: S1 > S2 > S3 > S4 by unsigned integer
  comparison.
5. **Exhaustive**: every Kline with candidates is S1, S2, or S3.
6. **S1 is trivial**: all nodes match → distance = 0, no function call needed.

## Code Locations

| Concept | File | Symbol |
| ------- | ---- | ------ |
| Constants (`D_MAX`, `MASK64`) | `src/kalvin/agent.py` | module-level |
| Routing | `src/kalvin/agent.py` | `Agent._route()` |
| Significance inversion | `src/kalvin/agent.py` | `Cogitator._process()` |
| Distance computation | `src/kalvin/model.py` | `Model.expand()` |
