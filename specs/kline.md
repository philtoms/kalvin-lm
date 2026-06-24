# Kline Specification

## Overview

A Kline is the fundamental unit of Kalvin's memory. It is an identified,
ordered sequence of zero or more nodes.

## Definition

A Kline consists of:

| Field     | Type               | Description                        |
| --------- | ------------------ | ---------------------------------- |
| signature | uint64             | Identity key.                      |
| nodes     | sequence of uint64 | Zero or more child nodes. Ordered. |

### Nodes

- A node is a 64-bit unsigned integer.
- Nodes are **opaque** — the kline does not inspect or interpret node values.
- Node order is significant. `[A, B]` and `[B, A]` are different klines.
- `nodes` may be empty (zero nodes).

### Signature

- The signature is a 64-bit unsigned integer that identifies the kline.
- Signatures are uint64 values occupying the kline's head position. See
  the @signature spec for the concept; creation and matching are defined in
  the @signifier spec.
- It is assigned at construction time.
- Signatures are not inherently unique. Duplication handling is a model
  responsibility.

## Construction

A Kline is constructed from a signature and a sequence of nodes:

```
Kline(signature, nodes)
```

- `signature` — required, uint64.
- `nodes` — required, zero or more uint64 values.

Implementations may accept multiple input representations for `nodes`
(single value, empty, list) provided the result is semantically identical:
a sequence of zero or more nodes.

## Equality

Two Klines are equal if and only if:

1. Their signatures are equal, **and**
2. Their node sequences are equal (same length, same order, same values).

## Operations

### Node access

```
kline.nodes → sequence of uint64
```

Returns the node sequence. An empty kline returns an empty sequence.

### Node count

```
len(kline.nodes) → int ≥ 0
```

The number of nodes. Equivalent to `len(kline.nodes)`.

## Structural Predicates

A kline's structural kind is determined by its signature and nodes alone
(no model state). Two predicates capture the kinds relevant to rationalisation:

- **`is_identity(kline)`** — `True` for the empty form `{S: []}` and the
  self-referential form `{S: [S]}` (sole node equals signature). Both carry
  no decomposition. The self-referential form is identity *by definition*
  and overrules any canon classification (see @CONTEXT.md §Identity).
- **`is_canon(kline)`** — `True` when the kline is neither identity nor
  self-referential AND `signature == make_signature(nodes)`.

These live with the KLine because they are structural properties; the model
and significance modules consume them.

| ID    | Criterion                                                                  |
| ----- | -------------------------------------------------------------------------- |
| KL-20 | `is_identity({S: []})` → True                                              |
| KL-21 | `is_identity({S: [S]})` → True (self-referential)                          |
| KL-22 | `is_identity({S: [A]})` (A ≠ S) → False                                    |
| KL-23 | `is_canon({S: [A, B]})` where `S == A\|B` and `S` not in nodes → True       |
| KL-24 | `is_canon({S: [S]})` → False (self-referential is identity, not canon)     |
| KL-25 | `is_canon({S: []})` → False (identity)                                     |

## What a Kline is Not

The following are explicitly **out of scope** for this spec:

- **Significance.** Significance is an assessment carried on a KValue, the
  unit of exchange (@kvalue spec). A Kline (the objective structure stored in
  memory) does not carry significance, compute it, or encode it in its
  signature. Significance is re-derived from structure on retrieval.
- **Bitwise matching.** AND/OR operations on signatures are model-level
  concerns, not kline operations.
- **Debug metadata.** Labels, source text, or other diagnostic information
  is implementation-level.

## Dependencies

The kline spec is self-contained with respect to node classification.

## Test Matrix

| ID    | Criterion                                                                     | Origin ref |
| ----- | ----------------------------------------------------------------------------- | ---------- |
| KL-1  | Construction with empty nodes produces empty list: `KLine(5, []).nodes == []` | —          |
| KL-2  | Construction with single int wraps into list: `KLine(5, 3).nodes == [3]`      | —          |
| KL-3  | Construction with list preserves list: `KLine(5, [1,2]).nodes == [1,2]`       | —          |
| KL-4  | Equality: same signature + same nodes → equal                                 | —          |
| KL-5  | Inequality: different signatures → not equal                                  | —          |
| KL-6  | Inequality: different node sequences → not equal                              | —          |
| KL-7  | Hash consistency: equal KLines produce equal hashes                           | —          |
| KL-11 | `len()` returns node count                                                    | —          |

## Referenced By

- **Significance** (@significance spec) — compares query and candidate Klines.
- **Model** (@model spec) — stores and retrieves Klines by signature.
- **Agent** (@agent spec) — encodes input into Klines, retrieves candidates.
