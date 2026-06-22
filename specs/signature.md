# Signature Specification

## Overview

A signature is the 64-bit unsigned integer occupying a kline's head
position. It identifies the kline for lookup and, when placed as a node in
another kline, forms an edge in the model. Signatures serve two roles —
head (identity) and node (selection) — and bottom out at value 0 (identity).

How signatures are *produced* from nodes and how they are *compared* for
overlap is the **signature algebra**, defined in the @signifier spec. This
spec defines the signature as a value and its roles in the model; it does
not define the algebra.

## Dependencies

### Kline (@kline spec)

- A kline's `signature` field is a signature (uint64).
- A kline's nodes may contain signatures (node values that reference other
  klines, forming graph edges).

### Tokenizer (@tokenizer spec)

- **Typed-node layout** — nodes are packed as
  `(sig_word << 32) | bpe_token_id`. A signature is built from such nodes.
  See @tokenizer for the layout.

### Signifier (@signifier spec)

- Signatures are produced by `make_signature` and compared by `signifies` —
  both defined in the @signifier spec. The Signifier is the sole authority
  for signature bit-operations; Kalvin treats signatures as opaque values.

## Definition

A **signature** is a uint64 value. It is identified by its role — the
`signature` field of a kline — not by a bit-pattern test. There is no
`is_signature` predicate; any uint64 value may serve as a signature.

### Roles

A signature serves two distinct roles in the model:

**1. Kline Head (Identity)**

When placed in a kline's `signature` field, a signature identifies the
kline. It acts as the primary lookup key — the model indexes klines by
signature. The same signature may identify multiple klines (signatures are
not inherently unique).

**2. Kline Node (Selection)**

When a signature appears as a node in another kline's node sequence, it
forms an **edge** in the model. If a node value equals the signature of
another kline, the model can resolve the node to that kline. This enables
compositional hierarchies: a kline's nodes reference other klines.

A node used as a kline node selects all klines whose signatures overlap
with it (see @signifier spec, `signifies`).

## Creation

Signatures are produced by the Signifier — see @signifier spec
(`make_signature`). This spec does not define the reduction.

## Well-known Values

| Value | Meaning                                              |
| ----- | ---------------------------------------------------- |
| 0     | No nodes at all. An identity kline.                    |

A signature of 0 means the kline has no nodes — it is an **identity**
kline carrying no structural identity. It cannot be found via overlap
matching (`signifies(0, x) == False`; see @signifier spec).

## What a Signature is Not

The following are explicitly **out of scope** for this spec:

- **Creation and matching.** How signatures are produced
  (`make_signature`) and compared for overlap (`signifies`) is defined in
  the @signifier spec.
- **Significance computation.** How signatures contribute to distance and
  significance values is defined in the @significance / @model specs.
- **Storage and indexing.** How the model indexes and retrieves klines by
  signature is defined in the @model spec.
- **Kline construction.** How signatures are assigned to klines at build
  time is defined in the @kline spec. How the agent prepares a kline's
  signature during rationalisation is defined in the @agent spec.

## Test Matrix

The signature value is exercised through the algebra; see the test matrix
in the @signifier spec (SIG-1 … SIG-16).

## Referenced By

- **Kline** (@kline spec) — the signature field is a signature as defined
  here.
- **Signifier** (@signifier spec) — produces and compares signatures.
- **Model** (@model spec) — indexes klines by signature, provides candidate
  retrieval via overlap matching (@signifier).
- **Agent** (@agent spec) — creates signatures during the prepare phase of
  rationalisation.
- **Significance** (@significance spec) — signatures are the basis for
  candidate retrieval but significance computation is separate.
