# Significance Specification

## Overview

Significance is a metric that describes how well a candidate Kline answers a
query Kline. It is computed as the inverse of a distance in a metric space.
All calculation is performed in distance space and converted to significance by
bitwise NOT.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### Kline (@kline spec)
- A Kline is an ordered sequence of nodes.
- Nodes are opaque; significance passes them to model functions without
  inspection.

### Model (@model spec)
- The model provides three functions (see Model API section).
- The model may be stateful: two calls with identical arguments may return
  different results if the model has been updated between calls. Significance
  does not manage this state — it is the model's responsibility.

### Agent (@agent spec)
- The agent encodes input into a query Kline Q.
- The agent retrieves candidate Klines {C₁ … Cₙ}.
- The agent consumes significance results downstream.
- Both encoding and candidate retrieval are upstream of significance and
  defined in the agent spec.

## Pipeline

```
Preconditions (agent responsibility):
  - Query Kline Q is provided
  - Candidate Klines {C₁ … Cₙ} are provided (possibly empty)

Significance computation:

    No candidates? → return single S4 result (candidate = None)

    For each candidate Cᵢ:
        Calculate significance(Q, Cᵢ)
          = per-node significance test → route → distance → inversion

Output: list of (candidate, SignificanceResult) tuples
  - With candidates: one tuple per candidate
  - No candidates: single (None, S4_result) tuple
```

The pipeline handles all significance levels including S4. Callers pass
candidates directly without pre-testing for empty.

## Per-node Significance

For each node in Q, the model answers a binary question: **does this node
achieve S1 significance against Cᵢ or not?** Two nodes at S1 significance
represent a perfect match at that point in the kline.

Nodes do not retain individual significance values — they contribute to the
kline-level result. Because the algorithm is **pessimistic**, overall kline
significance can only drop below the ideal. Even if some nodes achieve S1, the
presence of any node that does not pulls the overall level down.

The S1 test is a model function (`is_s1`). Significance routing depends only
on its boolean return value, not on how it computes it. Semantics of the test
are defined in the model spec.

## Routing to Distance Function

The pattern of per-node significance across all nodes routes to the appropriate
distance function:

```
s1_count = number of nodes in Q at S1 significance vs Cᵢ

if no candidates:                 distance = 0xFFFF_FFFF_FFFF_FFFF   → S4
elif s1_count == len(Q.nodes):    distance = 0                        → S1
elif s1_count > 0:                distance = model.s2_distance(Q, Cᵢ) → S2
elif s1_count == 0:               distance = model.s3_distance(Q, Cᵢ) → S3
```

| Node significance pattern | Routes to                          | Result |
| ------------------------- | ---------------------------------- | ------ |
| All nodes S1              | distance = 0 (no function call)    | S1     |
| Some nodes S1, some not   | `model.s2_distance(Q, Cᵢ)`         | S2     |
| No nodes S1               | `model.s3_distance(Q, Cᵢ)`         | S3     |
| No candidates             | distance = MAX_UINT64 (no call)    | S4     |

Only S2 and S3 routes call the model API.

This is exhaustive. Every Kline with at least one candidate is S1, S2, or S3.

The overall level (S1–S4) is returned alongside the significance value for use
by the caller.

### Agency Rationalisation

The routing levels correspond to structural agency categories defined in
the overview:

| Route | Agency category | Structural reason                                              |
| ----- | --------------- | -------------------------------------------------------------- |
| S1    | Canonical       | Signature fully represents nodes; all nodes match              |
| S1    | Countersigned   | Mutual cross-reference discovered via cogitation               |
| S2    | Underfitting    | Signature contains more information than nodes confirm         |
| S2    | Overfitting     | Signature contains less information than nodes provide         |
| S3    | Connotational   | Nodes unrelated to signature; association without composition  |
| S4    | Unsigned        | No candidates found; no information content                    |

S4 is produced by the pipeline itself (not the caller) when no candidates
are provided.

Countersigned S1 is not detected by the significance pipeline itself — it
is a latent relationship discovered during cogitation (see @agent spec).
The significance pipeline detects canonical S1 (all nodes match).

## Significance Representation

Significance is a **64-bit unsigned integer**, calculated as `~distance`.

### Arithmetic note

All distance and significance values are unsigned 64-bit integers. The `~`
operator denotes bitwise NOT on a uint64. Languages with signed arithmetic
must mask accordingly:

```
significance = (~distance) & 0xFFFF_FFFF_FFFF_FFFF
```

### Distance range

Distance is a single 64-bit unsigned integer. A boundary value `D_boundary`
partitions the range:

```
  0                 D_boundary                  0xFFFF_FFFF_FFFF_FFFF
  ├─────── S2 ──────────┼──────────── S3 ─────────────┤
```

- S1: distance = 0 (all nodes match, no computation)
- S2: distance ∈ [1, D_boundary)
- S3: distance ∈ [D_boundary, 0xFFFF_FFFF_FFFF_FFFF)
- S4: distance = 0xFFFF_FFFF_FFFF_FFFF (no candidates, no computation)
- Arithmetic ordering is guaranteed: `S1 > S2 > S3 > S4` because `~distance`
  is monotonically decreasing.

### Values

```
S1 = 0xFFFF_FFFF_FFFF_FFFF   (all 64 bits set, zero distance)
S4 = 0x0000_0000_0000_0000   (all 64 bits clear, maximum distance)
```

### D_boundary

`D_boundary` is a **hyperparameter** that separates the S2 and S3 distance
ranges.

- Default: `0x8000_0000_0000_0000` (the midpoint, giving S2 and S3 equal
  range).
- Configurable at initialisation.

### Distance calculation

Distance is produced by a single call to the appropriate model function:

```
# Routed by per-node significance pattern
if all nodes S1:      distance = 0
if some nodes S1:     distance = model.s2_distance(Q, Cᵢ)
if no nodes S1:       distance = model.s3_distance(Q, Cᵢ)
```

The model function returns a 64-bit distance value:

- `s2_distance` must return a value in [1, D_boundary).
- `s3_distance` must return a value in [D_boundary, 0xFFFF_FFFF_FFFF_FFFF).

If a model function returns a value outside its required range, the result is
**clamped** to the nearest valid value.

### Conversion

```
significance = ~distance
```

### Worked examples

```
Q = [node_a, node_b, node_c]
C = candidate

node_a → S1
node_b → S1
node_c → not S1

s1_count = 2, total = 3  →  some S1 → S2 route

distance     = model.s2_distance(Q, C)  = 0x0000_0000_0000_0005
significance = ~distance = 0xFFFF_FFFF_FFFF_FFFA   → S2 level
```

```
Q = [node_a]
C = candidate

node_a → S1

s1_count = 1, total = 1  →  all S1 → S1 route

distance     = 0
significance = ~0 = 0xFFFF_FFFF_FFFF_FFFF   → S1 level
```

```
Q = [node_a, node_b]
C = candidate

node_a → not S1
node_b → not S1

s1_count = 0  →  no S1 → S3 route

distance     = model.s3_distance(Q, C)  = 0x0000_0000_FFFF_FFFF
significance = ~distance = 0xFFFF_FFFF_0000_0000   → S3 level
```

```
Q = [node_a, node_b]
(no candidates)

significance = 0x0000_0000_0000_0000   → S4 level (no computation)
```

## Model API

The model provides:

```
model.is_s1(node, C) → bool                # does this node achieve S1 vs C?
model.s2_distance(Q, C) → uint64           # distance when some nodes are S1
model.s3_distance(Q, C) → uint64           # distance when no nodes are S1
```

- `is_s1` — returns whether a node achieves S1 match against candidate C.
  Semantics defined in the **model spec**.
- `s2_distance` — returns a distance in [1, D_boundary). Semantics defined in
  the **model spec**.
- `s3_distance` — returns a distance in [D_boundary, 0xFFFF_FFFF_FFFF_FFFF).
  Semantics defined in the **model spec**.
- Distance values are clamped to their required range if the model returns
  out-of-bounds.

## Summary of Properties

1. **Inverted metric**: significance = ~distance. Higher is more significant.
2. **Pessimistic**: overall significance can only drop below the per-node ideal.
3. **Routing-based**: per-node significance pattern routes to the appropriate
   distance function.
4. **Arithmetically comparable**: S1 > S2 > S3 > S4 by unsigned integer
   comparison.
5. **State-aware**: significance may change if the model is updated between
   calls. State management is the model's responsibility.
6. **Exhaustive**: every Kline with candidates is S1, S2, or S3.
7. **Hyperparameter boundary**: `D_boundary` separates S2 and S3 distance
   ranges. Default is the midpoint.
8. **S1 is trivial**: all nodes S1 → distance = 0, no function call needed.
