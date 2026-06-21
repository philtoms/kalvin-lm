# Signature Specification

## Overview

A signature is a 64-bit unsigned integer produced by `make_signature`. It is
a plain OR-reduction of raw unmasked node values — no branching, no masking,
no special cases. Signatures serve two roles in the model:

1. **As kline head** — the `signature` field of a kline identifies it and
   enables lookup.
2. **As kline node** — a node value that references or selects other klines,
   forming the edges of the model.

Every node contributes its full value to the OR-reduction regardless of type.

## Dependencies

This spec depends on the following concepts, defined elsewhere:

### Kline (@kline spec)

- A kline's `signature` field is a signature (uint64 produced by
  `make_signature`).
- A kline's nodes may contain signatures (node values that reference other
  klines, forming graph edges).

### Tokenizer (@tokenizer spec)

- `is_nlp_node(node) → bool` — determines whether a node is an NLP-BPE
  node (non-zero high 32 bits), distinguishing tokenizer-produced nodes
  from non-tokenized uint64 values (e.g. node value 0 for identity klines,
  or signatures used as graph-edge references). Available for
  compiler/kline_display text rendering — deciding which nodes can be
  BPE-decoded versus treated as opaque references. Not used by
  `make_signature`.
- **NLP-BPE node layout** — nodes are packed as
  `(nlp_type32 << 32) | bpe_token_id`: the upper 32 bits carry NLP type
  (POS + DEP + MORPH) and the lower 32 carry the BPE token id. `signifies`
  masks off the lower (BPE) 32 bits and compares only the upper (type)
  half (see §Bitwise AND Matching).

## Definition

A **signature** is a uint64 value produced by `make_signature`. A signature
is identified by its role (the `signature` field of a kline), not by a
bit-pattern test. There is no `is_signature` predicate — any uint64 value
may serve as a signature.

### Roles

A signature serves two distinct roles in the model:

**1. Kline Head (Identity)**

When placed in a kline's `signature` field, a signature identifies the
kline. It acts as the primary lookup key — the model indexes klines by
signature. The same signature may identify multiple klines (signatures are
not inherently unique).

**2. Kline Node (Selection)**

When a signature appears as a node in another kline's node sequence, it
forms an **edge** in the model. If a node value equals the
signature of another kline, the model can resolve the node to that kline.
This enables compositional hierarchies: a kline's nodes reference other
klines.

A node used as a kline node selects all klines whose signatures overlap
with it in the upper (NLP-type) 32 bits (see §Bitwise AND Matching).

## Creation

### make_signature

```
make_signature(nodes: sequence of uint64) → uint64
```

Produce a signature from a sequence of nodes by OR-reducing all raw node
values:

```
sig = 0
for node in nodes:
    sig |= node
return sig
```

Every node contributes its full value. No masking, no branching, no special
cases. The result is deterministic and commutative.

### Properties

| Property      | Rule                                               |
| ------------- | --------------------------------------------------- |
| Deterministic | Same node set → same signature                     |
| Commutative   | Node order does not affect the result              |
| Lossy         | Order and multiplicity of nodes are lost           |
| Empty         | `make_signature([]) == 0`                          |
| Identity      | `make_signature([x]) == x` for any single node     |

### Well-known Values

| Value | Meaning                                              |
| ----- | ---------------------------------------------------- |
| 0     | No nodes at all. An identity kline.                    |

A signature of 0 means the kline has no nodes — it is an **identity** kline
carries no structural identity. It cannot be found via bitwise AND matching
(since `x & 0 == 0` for all x).

## Bitwise AND Matching

Signatures support bitwise AND matching as a fast, approximate similarity
test over **NLP-type bits only**. The NLP-BPE node layout (see @tokenizer
spec) packs NLP type information (POS + DEP + MORPH) into the upper 32
bits and the BPE token ID into the lower 32 bits. `signifies` masks off
the lower (BPE) 32 bits so two klines are compared by NLP-type overlap,
not by token-ID collision:

```
TYPE_MASK = 0xFFFF_FFFF_0000_0000  # upper 32 bits = NLP type
signifies(a, b) → (a & b & TYPE_MASK) != 0
```

Two signatures that share at least one set type bit are considered
overlapping. This is the basis for candidate retrieval in the
rationalisation pipeline:

```
candidates = model.where(k => (k.signature & query.signature & TYPE_MASK) != 0)
```

Properties:

- **Necessary, not sufficient** — type overlap is a pre-filter. The
  significance pipeline determines actual relevance.
- **Commutative** — `signifies(a, b) == signifies(b, a)`.
- **False positives** — overlapping type bits may not be semantically
  related.
- **False negatives** — non-overlapping type bits are guaranteed
  irrelevant (no shared type bits).
- **Vacuous for 0** — a signature of 0 never signifies anything.
- **Type-only** — values whose set bits fall entirely in the lower 32
  (e.g. a raw BPE token id with no NLP type) never signify anything,
  because their upper 32 bits are zero.

Masking applies only to `signifies`. `make_signature` still OR-reduces the
full raw node values; a signature therefore carries both halves, and the
lower (BPE) bits remain available for decoding and exact-match resolution.

## Test Matrix

| ID    | Criterion                                                    | Origin ref |
| ----- | ------------------------------------------------------------ | ---------- |
| SIG-1 | `make_signature([]) == 0` (empty → identity)                  | — |
| SIG-4 | `make_signature([x]) == x` (identity)                        | — |
| SIG-6 | `make_signature([A, B]) == A \| B` (commutative, OR-reduce)   | — |
| SIG-7 | `signifies(0, anything) == False` (vacuous for 0)             | — |
| SIG-9 | `signifies(T(0b110), T(0b010)) == True` (overlapping type bits) | — |
| SIG-10 | `signifies(T(0b100), T(0b010)) == False` (no overlapping type bits) | — |
| SIG-14 | `make_signature([0b10, 0b100]) == 0b110` (OR-reduction of two distinct node values) | NLP |
| SIG-15 | `signifies(0b110, 0b010) == False` (lower/BPE bits masked off) | — |
| SIG-16 | `signifies(T(0b110) \| 1, T(0b010) \| 2) == True` (type overlap beats differing BPE ids) | — |

`T(x)` denotes NLP-type bits `x` packed into the upper 32 (`x << 32`),
with the lower 32 (BPE token id) zero. See the @tokenizer spec for the
full NLP-BPE node layout.

## What a Signature is Not

The following are explicitly **out of scope** for this spec:

- **Node encoding.** How nodes are constructed (character maps, type
  prefixes, etc.) is defined in the @tokenizer spec.
- **Significance computation.** How signatures contribute to distance and
  significance values is defined in the @significance spec. The inversion
  `(~distance) & MASK64` is performed in `agent.py`.
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
