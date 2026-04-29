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
- The model provides distance functions (see Model API section).
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
        Calculate significance(Q, Cᵢ):
          1. Route   — per-node match test → determine route (S1/S2/S3)
          2. Distance — call appropriate distance function (or 0 for S1)
          3. Invert  — significance = ~distance

Output: list of (candidate, SignificanceResult) tuples
  - With candidates: one tuple per candidate
  - No candidates: single (None, S4_result) tuple
```

The pipeline handles all significance levels including S4. Callers pass
candidates directly without pre-testing for empty.

## Step 1: Route

Routing is the first step of the significance pipeline. It is a simple,
independent algorithm that determines which distance function (if any) to
call.

For each node in Q, a binary test is applied: **does this node exist in the
candidate?** The test checks whether the node value appears in the candidate's
node sequence. This is a straightforward membership test — it does not
involve significance, distance, or any model function.

The matched nodes are counted and the result is routed:

- **All nodes match** → S1 route (distance = 0, no function call)
- **Some nodes match** → S2 route (calls `model.s2_distance`)
- **No nodes match** → S3 route (calls `model.s3_distance`)

Routing is **pessimistic**: the presence of any node in Q that doesn't exist
in the candidate prevents an S1 result. Only S2 and S3 routes invoke model
distance functions.

## Step 2: Distance

The routing result determines the distance computation:

```
match_count = number of nodes in Q that exist in Cᵢ's node sequence

if no candidates:                  distance = 0xFFFF_FFFF_FFFF_FFFF   → S4
elif match_count == len(Q.nodes):  distance = 0                        → S1
elif match_count > 0:              distance = model.s2_distance(Q, Cᵢ) → S2
elif match_count == 0:             distance = model.s3_distance(Q, Cᵢ) → S3
```

| Routing result              | Distance computation             | Result |
| --------------------------- | -------------------------------- | ------ |
| All nodes match             | distance = 0 (no function call)  | S1     |
| Some nodes match, some not  | `model.s2_distance(Q, Cᵢ)`       | S2     |
| No nodes match              | `model.s3_distance(Q, Cᵢ)`       | S3     |
| No candidates               | distance = MAX_UINT64 (no call)  | S4     |

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

## Step 3: Invert

The final step converts distance to significance by bitwise NOT:

```
significance = ~distance
```

Significance is a **64-bit unsigned integer**.

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
# Routed by per-node match count
if all nodes match:   distance = 0
if some nodes match:  distance = model.s2_distance(Q, Cᵢ)
if no nodes match:    distance = model.s3_distance(Q, Cᵢ)
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

node_a → match (exists in C)
node_b → match (exists in C)
node_c → no match

match_count = 2, total = 3  →  S2 route

distance     = model.s2_distance(Q, C)  = 0x0000_0000_0000_0005
significance = ~distance = 0xFFFF_FFFF_FFFF_FFFA   → S2 level
```

```
Q = [node_a]
C = candidate

node_a → match (exists in C)

match_count = 1, total = 1  →  S1 route

distance     = 0
significance = ~0 = 0xFFFF_FFFF_FFFF_FFFF   → S1 level
```

```
Q = [node_a, node_b]
C = candidate

node_a → no match
node_b → no match

match_count = 0  →  S3 route

distance     = model.s3_distance(Q, C)  = 0x0000_0000_FFFF_FFFF
significance = ~distance = 0xFFFF_FFFF_0000_0000   → S3 level
```

```
Q = [node_a, node_b]
(no candidates)

significance = 0x0000_0000_0000_0000   → S4 level (no computation)
```

## Model API

Routing (Step 1) is performed entirely within the significance pipeline using
simple node-membership testing. It does not call any model function.

Steps 2 and 3 consume the following model functions:

```
model.s2_distance(Q, C) → uint64           # distance when some nodes match
model.s3_distance(Q, C) → uint64           # distance when no nodes match
```

- `s2_distance` — returns a distance in [1, D_boundary). Semantics defined in
  the **model spec**.
- `s3_distance` — returns a distance in [D_boundary, 0xFFFF_FFFF_FFFF_FFFF).
  Semantics defined in the **model spec**.
- Distance values are clamped to their required range if the model returns
  out-of-bounds.

## Summary of Properties

1. **Inverted metric**: significance = ~distance. Higher is more significant.
2. **Three-step pipeline**: Route → distance → invert. Each step is independent.
3. **Routing is self-contained**: routing uses simple node-membership testing
   and does not call any model function.
4. **Pessimistic**: the presence of any unmatched node prevents S1.
5. **Arithmetically comparable**: S1 > S2 > S3 > S4 by unsigned integer
   comparison.
6. **State-aware**: significance may change if the model is updated between
   calls. State management is the model's responsibility.
7. **Exhaustive**: every Kline with candidates is S1, S2, or S3.
8. **Hyperparameter boundary**: `D_boundary` separates S2 and S3 distance
   ranges. Default is the midpoint.
9. **S1 is trivial**: all nodes match → distance = 0, no function call needed.
