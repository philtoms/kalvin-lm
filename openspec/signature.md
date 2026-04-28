# Signature Specification

## Overview

A signature is a 64-bit unsigned integer whose bit 0 is clear. Signatures serve two roles in
the knowledge graph:

1. **As kline head** — the `signature` field of a kline identifies it and
   enables lookup.
2. **As kline node** — a non-literal node value that references or selects
   other klines, forming the edges of the knowledge graph.

Signatures are produced by OR-reduction of non-literal nodes. Because
non-literal nodes have bit 0 clear, the result also has bit 0 clear — the
operation preserves the signature property.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### Tokenizer (@tokenizer spec)

- `is_literal(node) → bool` — determines whether a node is literal (bit 0
  set) or non-literal (bit 0 clear). Signature creation uses this to filter
  which nodes contribute.
- Tokenizer encoding determines what non-literal nodes look like (bit
  patterns, character maps, type prefixes). The signature spec treats nodes
  as opaque beyond the literal test.

### Kline (@kline spec)

- A kline's `signature` field is a signature (uint64 with bit 0 clear).
- A kline's nodes may contain signatures (non-literal node values that
  reference other klines).

## Definition

A **signature** is a uint64 value whose bit 0 is clear:

```
is_signature(value) → (value & 1) == 0
```

This is the complement of the literal test. Every node is either a literal
(bit 0 set) or a signature (bit 0 clear):

| Bit 0 | Kind      | Role                                   |
| ----- | --------- | -------------------------------------- |
| 1     | Literal   | Exact token — preserves identity/order |
| 0     | Signature | Structural — identifies or selects     |

### Roles

A signature serves two distinct roles in the knowledge graph:

**1. Kline Head (Identity)**

When placed in a kline's `signature` field, a signature identifies the
kline. It acts as the primary lookup key — the model indexes klines by
signature. The same signature may identify multiple klines (signatures are
not inherently unique).

**2. Kline Node (Selection)**

When a signature appears as a node in another kline's node sequence, it
forms an **edge** in the knowledge graph. If a node value equals the
signature of another kline, the model can resolve the node to that kline.
This enables compositional hierarchies: a kline's nodes reference other
klines.

A single non-literal node is itself a signature. When used as a kline
node, it selects all klines whose signatures overlap with it (bitwise
AND ≠ 0).

## Creation

### make_signature

```
make_signature(nodes: sequence of uint64) → uint64
```

Produce a signature by OR-reduction of all non-literal nodes in the
sequence.

```
sig = 0
for node in nodes:
    if not is_literal(node):
        sig |= node
return sig
```

Because every non-literal node has bit 0 clear, the OR-reduction of
non-literal nodes also has bit 0 clear. The result is therefore itself a
signature.

### Properties

| Property        | Rule                                                  |
| --------------- | ----------------------------------------------------- |
| Deterministic   | Same node set → same signature                        |
| Commutative     | Node order does not affect the result                 |
| Lossy           | Order and multiplicity are lost                       |
| Empty           | `make_signature([]) == 0`                             |
| Identity        | `make_signature([node]) == node` for non-literal node |
| Literal-exclude | Literal nodes do not contribute                       |
| Closure         | Result has bit 0 clear (is a signature)               |

### Well-known Values

| Value | Meaning                                    |
| ----- | ------------------------------------------ |
| 0     | No non-literal content. An unsigned kline. |

A signature of 0 means the kline has no non-literal nodes (either empty, or
all-literal). Such a kline carries no structural identity — it cannot be
found via bitwise AND matching (since `x & 0 == 0` for all x).

## Bitwise AND Matching

Signatures support bitwise AND matching as a fast, approximate similarity
test:

```
signifies(a, b) → (a & b) != 0
```

Two signatures that share at least one set bit are considered overlapping.
This is the basis for candidate retrieval in the rationalisation pipeline:

```
candidates = model.where(k => (k.signature & query.signature) != 0)
```

Properties:

- **Necessary, not sufficient** — overlap is a pre-filter. The significance
  pipeline determines actual relevance.
- **Commutative** — `signifies(a, b) == signifies(b, a)`.
- **False positives** — overlapping signatures may not be semantically
  related.
- **False negatives** — non-overlapping signatures are guaranteed
  irrelevant (no shared bits).
- **Vacuous for 0** — a signature of 0 never signifies anything.

## What a Signature is Not

The following are explicitly **out of scope** for this spec:

- **Node encoding.** How non-literal nodes are constructed (character maps,
  type prefixes, etc.) is defined in the @tokenizer spec.
- **Literal test implementation.** `is_literal` is defined in the
  @tokenizer spec. This spec depends on it but does not define it.
- **Significance computation.** How signatures contribute to distance and
  significance values is defined in the @significance spec.
- **Storage and indexing.** How the model indexes and retrieves klines by
  signature is defined in the @model spec.
- **Kline construction.** How signatures are assigned to klines at build
  time is defined in the @kline spec. How the agent prepares a kline's
  signature during rationalisation is defined in the @agent spec.

## Referenced By

- **Kline** (@kline spec) — the signature field is a signature as defined
  here.
- **Model** (@model spec) — indexes klines by signature, provides candidate
  retrieval via bitwise AND matching.
- **Agent** (@agent spec) — creates signatures during the prepare phase of
  rationalisation.
- **Significance** (@significance spec) — signatures are the basis for
  candidate retrieval but significance computation is separate.
