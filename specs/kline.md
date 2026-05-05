# Kline Specification

## Overview

A Kline is the fundamental unit of the knowledge graph. It is an identified,
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

### Literal

- A kline is **literal** if every one of its nodes is a literal token
  (per `is_literal`, defined below). A kline is **non-literal** otherwise.
- Literal status is not stored — it is computed from nodes on demand.
- An empty kline (zero nodes) is non-literal.
- The model uses `is_literal()` to determine deduplication behaviour
  (see @model spec).

### Signature

- The signature is a 64-bit unsigned integer that identifies the kline.
- Signatures are uint64 values produced by `make_signature`. See the
  @signature spec for the full definition, including creation and properties.
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

### Is literal

```
kline.is_literal() → bool
```

Returns whether every node in this kline is a literal token.
Equivalent to `all(is_literal(node) for node in kline.nodes)`.
An empty kline returns `false`.

### `is_literal(node: int) → bool`

A standalone function that tests whether a single node is a literal token.
This is a bit-layout test, not a tokenizer method:

```
(node & 0xFFFFFFFF) == 0xFFFFFFFF
```

The lower 32 bits being all set is the **literal mask**. This distinguishes
literal nodes from packed nodes (bit 0 clear) and from signature values
(bit 0 may be set but the lower 32 bits are not all 1s).

This function is the single authority for the literal/non-literal distinction.
All components that need to classify nodes use this function directly — it
is not injected from the tokenizer.

## What a Kline is Not

The following are explicitly **out of scope** for this spec:

- **Significance.** Significance is a computed metric defined in the
  significance spec. A Kline does not carry significance, compute it,
  or encode it in its signature.
- **Bitwise matching.** AND/OR operations on signatures are model-level
  concerns, not kline operations.
- **Debug metadata.** Labels, source text, or other diagnostic information
  is implementation-level.
- **Deduplication.** The computed literal property signals deduplication
  eligibility, but the model owns deduplication logic (@model spec).

## Dependencies

The kline spec is self-contained with respect to node classification.
`is_literal(node) → bool` is defined here as a standalone bit-layout test.
It is not imported from any other spec.

## Referenced By

- **Significance** (@significance spec) — compares query and candidate Klines.
- **Model** (@model spec) — stores and retrieves Klines by signature.
- **Agent** (@agent spec) — encodes input into Klines, retrieves candidates.
