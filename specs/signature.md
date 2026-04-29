# Signature Specification

## Overview

A signature is a 64-bit unsigned integer produced by `make_signature`. Bit 0
is the **literal-content flag** — it indicates whether the kline contains
literal nodes. Signatures serve two roles in the knowledge graph:

1. **As kline head** — the `signature` field of a kline identifies it and
   enables lookup.
2. **As kline node** — a node value that references or selects other klines,
   forming the edges of the knowledge graph.

Signatures are produced by OR-reduction of all nodes. Non-literal nodes
contribute their full value. Literal nodes contribute bit 0 only. Because
non-literal nodes have bit 0 clear, the OR-reduction sets bit 0 if and only
if literal content is present.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### Tokenizer (@tokenizer spec)

- `is_literal(node) → bool` — determines whether a node is a literal token.
  Signature creation uses this to determine how each node contributes:
  non-literal nodes contribute their full value; literal nodes contribute
  bit 0 only. The `is_literal` test is defined by the tokenizer and varies
  by encoding — see @tokenizer spec.
- Tokenizer encoding determines what non-literal and literal nodes look like.
  The signature spec treats nodes as opaque beyond the `is_literal` test.

### Kline (@kline spec)

- A kline's `signature` field is a signature (uint64 produced by
  `make_signature`).
- A kline's nodes may contain signatures (node values that reference other
  klines, forming graph edges).

## Definition

A **signature** is a uint64 value produced by `make_signature`. A signature
is identified by its role (the `signature` field of a kline), not by a
bit-pattern test. There is no `is_signature` predicate — any uint64 value
may serve as a signature.

### Bit 0: The Literal-Content Flag

Bit 0 of a signature indicates whether the kline contains literal nodes:

| Bit 0 | Meaning                        |
| ----- | ------------------------------ |
| 1     | Kline contains literal nodes   |
| 0     | Kline has no literal nodes     |

This flag is set by `make_signature` when literal nodes are present. It is
not a discriminator between node types — it is a property of the signature
itself.

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

A node used as a kline node selects all klines whose signatures overlap
with it (bitwise AND ≠ 0).

## Creation

### make_signature

```
make_signature(nodes: sequence of uint64) → uint64
```

Produce a signature from a sequence of nodes. Non-literal nodes contribute
their full value via OR-reduction. Literal nodes contribute bit 0 only
(the literal-content flag), indicating that the kline contains literal
content.

```
sig = 0
for node in nodes:
    if is_literal(node):
        sig |= 1                  # literal-content flag (bit 0)
    else:
        sig |= node                # non-literal contributes full value
return sig
```

Because non-literal nodes have bit 0 clear, the OR-reduction sets bit 0
if and only if literal content is present. The result is deterministic and
commutative.

### The Literal-Content Flag (Bit 0)

Bit 0 is the **literal-content flag**. When any literal node is present in
the sequence, bit 0 is set in the signature. This ensures that klines
containing only literal nodes still produce a non-zero signature (`1`),
making them discoverable via bitwise AND matching and allowing the
canonical test (`signature == make_signature(nodes)`) to succeed for
all-literal klines.

Consequences:

- An all-literal kline (e.g. `[literal_A, literal_B]`) produces signature
  `1` — a valid, non-zero signature.
- An empty kline (`[]`) still produces signature `0` — unsigned.
- A kline mixing literal and non-literal nodes has both bit 0 and the
  non-literal bits set.

### Properties

| Property          | Rule                                                  |
| ----------------- | ----------------------------------------------------- |
| Deterministic     | Same node set → same signature                        |
| Commutative       | Node order does not affect the result                 |
| Lossy             | Order and multiplicity of non-literal nodes are lost  |
| Empty             | `make_signature([]) == 0`                             |
| Identity          | `make_signature([node]) == node` for non-literal node |
| Literal-include   | Literal nodes contribute bit 0 (literal-content flag) |
| Closure           | Bit 0 set iff literal content present                 |

Note: literal nodes are lossy in a different sense than non-literal nodes.
Non-literal nodes lose order and multiplicity. Literal nodes lose identity
entirely — all literal nodes collectively contribute only bit 0, regardless
of how many there are or what values they hold.

### Well-known Values

| Value | Meaning                                              |
| ----- | ---------------------------------------------------- |
| 0     | No nodes at all. An unsigned kline.                   |
| 1     | Contains literal content only (no non-literal nodes). |

A signature of 0 means the kline has no nodes — it is **unsigned** and
carries no structural identity. It cannot be found via bitwise AND matching
(since `x & 0 == 0` for all x).

A signature of `1` means the kline contains only literal nodes. It has
structural identity (non-zero) and is discoverable via bitwise AND matching.
Bit 0 is set, reflecting the presence of literal content.

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

- **Node encoding.** How non-literal and literal nodes are constructed
  (character maps, type prefixes, literal masks, etc.) is defined in the
  @tokenizer spec.
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
