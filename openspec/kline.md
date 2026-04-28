# Kline Specification

## Overview

A Kline is the fundamental unit of the knowledge graph. It is an identified,
ordered sequence of zero or more nodes.

## Definition

A Kline consists of:

| Field     | Type               | Description                                   |
| --------- | ------------------ | --------------------------------------------- |
| signature | uint64             | Identity key.                                 |
| nodes     | sequence of uint64 | Zero or more child nodes. Ordered.            |
| literal   | bool               | Whether this kline represents an exact token. |

### Nodes

- A node is a 64-bit unsigned integer.
- Nodes are **opaque** — the kline does not inspect or interpret node values.
- Node order is significant. `[A, B]` and `[B, A]` are different klines.
- `nodes` may be empty (zero nodes).

### Literal

- A literal kline represents an exact, atomic token — e.g. a single word or
  symbol produced by the tokeniser.
- A non-literal kline is a composed structure — its nodes reference other
  klines.
- The literal flag is set at construction time and is immutable.
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

A Kline is constructed from a signature, a sequence of nodes, and a literal flag:

```
Kline(signature, nodes, literal)
```

- `signature` — required, uint64.
- `nodes` — required, zero or more uint64 values.
- `literal` — required, bool. Defaults to `false`.

Implementations may accept multiple input representations for `nodes`
(single value, empty, list) provided the result is semantically identical:
a sequence of zero or more nodes.

## Equality

Two Klines are equal if and only if:

1. Their signatures are equal, **and**
2. Their node sequences are equal (same length, same order, same values).

The `literal` flag does not participate in equality.

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

Returns whether this kline is literal.

## What a Kline is Not

The following are explicitly **out of scope** for this spec:

- **Significance.** Significance is a computed metric defined in the
  significance spec. A Kline does not carry significance, compute it,
  or encode it in its signature.
- **Bitwise matching.** AND/OR operations on signatures are model-level
  concerns, not kline operations.
- **Debug metadata.** Labels, source text, or other diagnostic information
  is implementation-level.
- **Deduplication.** The literal flag signals deduplication eligibility, but
  the model owns deduplication logic (@model spec).

## Dependencies

None. The kline is a leaf concept.

## Referenced By

- **Significance** (@significance spec) — compares query and candidate Klines.
- **Model** (@model spec) — stores and retrieves Klines by signature.
- **Agent** (@agent spec) — encodes input into Klines, retrieves candidates.
