# Signifier Specification

## Overview

The Signifier is the sole authority for how node values reduce to signatures
and how signatures relate. It is the peer of the Tokenizer: the Tokenizer
turns text into nodes, the Signifier turns nodes into signatures and relates
them. Kalvin receives both as independent, injected concerns and is agnostic
to either's internals.

The interface (`KSignifier`) specifies **role only**. It says nothing about
bit patterns, algebra, determinism, or node structure — it does not even
assume a signature is derived from bits. A concrete Signifier is free to
implement its operations by any means: the production `NLPSignifier` uses a
bit-algebra over a packed word, but a future Signifier might be probabilistic,
or a model trained on synthetic klines, with no bitwise reasoning at all.
Such a Signifier would violate none of this interface. Anything specific to
*how* a Signifier works — including every algebraic property of its output —
belongs to the concrete signifier (see §NLPSignifier), not the interface.

## Dependencies

### Tokenizer (@tokenizer spec)

The Signifier consumes the node values the Tokenizer produces. Beyond that
data flow, the interface imposes no relationship between them: whether a
concrete Signifier understands the Tokenizer's internal node layout is a
concrete-signifier concern (NLPSignifier does; see §NLPSignifier and
@nlp_tokenizer).

### Signature (@signature spec)

- Operates on **signature values** and **node values**, both defined in the
  @signature spec.
- The Signifier does not define what a signature *is* (its role as a kline
  head); it defines how signatures are *produced* and *compared*.

### Kline (@kline spec)

- Consumes the node values the Tokenizer produces and the signature values
  the Signifier produces.

## Definition

A **Signifier** is an injectable object that produces signature values from
node values and relates signature values to each other. It is the **sole
authority** over that production and relation: outside a Signifier
implementation, Kalvin treats node and signature values as **opaque** — it
stores them, compares them by equality, and passes them to the Signifier for
anything else.

The interface (`KSignifier`) fixes the operations' **role** only. How a
concrete Signifier fulfils that role — its algorithm, its algebraic
properties, even whether it operates on bits at all — is the concrete
signifier's own. The production concrete Signifier is `NLPSignifier` (see
§NLPSignifier); it is paired with the NLP Tokenizer to form the NLP
deployment bundle (see @nlp_tokenizer spec).

## Interface (`KSignifier`)

The interface specifies role and types. It is silent on mechanism and on
algebraic guarantees: it does not require the operations to be deterministic,
commutative, order-sensitive, or identity-preserving, and it does not
prescribe any signature value (including the empty-set signature). All such
characteristics are concrete-signifier properties (see §NLPSignifier).

### `make_signature(nodes: Sequence[uint64]) → uint64`

Produce the signature value that will occupy a kline's head position for the
given node sequence. The system indexes and retrieves klines by this value.

No property of the output is guaranteed by the interface — not determinism,
not order-independence, not the value returned for an empty sequence, not
that a single-node sequence maps to itself.

### `signifies(a: uint64, b: uint64) → bool`

A candidate-admission test used ahead of the significance computation in the
rationalisation pipeline: it reports whether two signature values are worth
evaluating as a candidate pair. It is a pre-filter, not the final relevance
determination — the significance computation decides actual relevance.

No property of the relation is guaranteed by the interface — not symmetry,
not that either result admits or excludes a candidate definitively.

### `residual(a: uint64, b: uint64) → <residual>`

The residual of signature *a* over signature *b*: a derived value
representing what *a* carries that *b* does not. Used to compute coverage
between a kline's signature and its nodes' signature — e.g. whether the
signature over- or under-claims its nodes.

The interface does not specify the residual's representation (it is opaque
to Kalvin), only that it can be passed back to the Signifier for emptiness
queries (see `classify_misfit` below). No algebraic property of the result
is guaranteed — not that it is a signature, not its value for equal inputs.

### `classify_misfit(signature: uint64, nodes: Sequence[uint64]) → tuple[bool, bool]`

Classify whether a signature faithfully covers its node set. Returns
`(underfit, overfit)`:

- `underfit` — the signature claims more than its nodes deliver.
- `overfit` — the nodes carry more than the signature captures.

Encapsulates the residual computation and its emptiness test, so callers
receive booleans and never inspect a residual value's representation. Used
by the misfit/expansion pipeline during rationalisation.

## NLPSignifier (Production Concrete)

`NLPSignifier` is the sole concrete Signifier and the production
implementation. It owns the NLP mechanism and all of its consequences. Under
the NLP interpretation (see @nlp_tokenizer spec), each node packs an NLP type
word in its upper 32 bits and a BPE token ID in its lower 32:

```
node = (sig_word << 32) | bpe_token_id      # sig_word = single type bit
TYPE_MASK = 0xFFFF_FFFF_0000_0000             # isolate the upper 32 bits
```

### Peer coupling with the NLP Tokenizer

NLPSignifier understands the NLP Tokenizer's node packing (it masks the upper
32 bits). This is the permitted coupling between the two NLP siblings: both
agree on the `sig_word` arrangement. It is a property of the NLP bundle,
not of the interface — a different Signifier need not understand any packing.

### `make_signature` — bitwise OR-reduce

OR-reduces the full 64-bit node values (`sig |= node` over the entire word).
Every node contributes its full value; the resulting signature accumulates
the `sig_word` words of all nodes.

### `signifies` — masked type-word overlap

Masks both values to the upper 32 bits (the NLP type word) and tests non-zero
AND:

```
signifies(a, b) → (a & b & TYPE_MASK) != 0
```

Under the NLP interpretation this tests whether two values share a type bit
(see @nlp_tokenizer for how `sig_word` bits are assigned). The BPE token IDs
(lower 32) are deliberately excluded, so two values are compared by type
overlap, not by token-ID collision.

### `residual` — masked type-word residual

Masks both values to the type word and returns the type-word bits *a* carries
that *b* does not:

```
residual(a, b) → (a & ~b) & TYPE_MASK
```

Masking to the type word is consistent with `signifies`: BPE-token-id
residuals are excluded, so the residual captures type-dimension claims, not
token-id differences.

### `classify_misfit` — masked residual classification

Computes `residual(signature, make_signature(nodes))` and
`residual(make_signature(nodes), signature)` and tests each for non-zero,
returning `(underfit, overfit)`. The emptiness test (`!= 0`) lives inside
this method — callers receive booleans and never inspect a residual value.

### Properties (NLPSignifier-specific)

Every property below is a consequence of the NLP bit-algebra. None of them is
required by the interface; each would be violated by a non-bit Signifier
(e.g. a probabilistic or learned one), which is why they live here.

**`make_signature`:**

- **Deterministic** — same node sequence → same signature.
- **Commutative** — node order does not affect the result (OR is order-free).
- **Lossy** — order and multiplicity are lost (`{A, B}` and `{A, A, B}`
  reduce identically).
- **Empty → 0** — `make_signature([]) == 0`. The value `0` is NLPSignifier's
  empty-set signature.
- **Identity** — `make_signature([x]) == x` (OR of a single value).

**`signifies`:**

- **Symmetric** — `signifies(a, b) == signifies(b, a)`.
- **Vacuous for 0** — `signifies(0, x) == False`, consistent with `0` being
  the empty signature.
- **Type-only** — values whose set bits fall entirely in the lower 32 (a raw
  BPE token id with no type word) never signify anything.
- **False positives** — overlapping type dimensions may not be semantically
  related.
- **Guaranteed false negatives** — non-overlapping type dimensions are
  guaranteed irrelevant, so `signifies == False` rules a candidate out.

**`residual`:**

- **Empty residual is `0`** — `residual(a, a) == 0` (a value has no residual
  over itself).
- **Type-only** — like `signifies`, masking excludes BPE-token-id residuals;
  two values differing only in BPE id have a zero residual.
- **Directional** — `residual(a, b) != residual(b, a)` in general; the two
  directions are the underfit/overfit residuals used by `classify_misfit`.

**`classify_misfit`:**

- **Encapsulates `residual` + `!= 0`** — the only operation Kalvin core uses
  over a residual value; callers receive `(underfit, overfit)` bools and
  never inspect the residual representation.

These constants, ranges, and the "type dimension" semantics are
NLP-deployment details; they do not appear in the interface and are not
visible to Kalvin.

## Opacity Invariant

**Kalvin treats node and signature values as opaque.** Outside a Signifier
implementation, Kalvin stores these values, compares them by equality, and
passes them to the Signifier for any reduction or relational question — and
does not otherwise interpret them.

In the NLP deployment this invariant is verified concretely by the absence of
bitwise operations (`&`, `|`, `~`, `<<`, `>>`) on node/signature values in
Kalvin core. (A non-bit Signifier deployment would verify opacity differently,
since its internals contain no bitwise operations to begin with.)

## Test Matrix

`T(x)` denotes type-word bits `x` packed into the upper 32 (`x << 32`), with
the lower 32 (BPE token id) zero. `T()` and the type/BPE distinction are
NLPSignifier specifics (see @nlp_tokenizer).

The interface has no algebraic test criteria — its contract is role-only.
Every criterion below is a property of NLPSignifier's bit-algebra:

| ID    | Criterion                                                    | Origin ref |
| ----- | ------------------------------------------------------------ | ---------- |
| SIG-1 | `make_signature([]) == 0` (empty → NLPSignifier's empty value) | — |
| SIG-4 | `make_signature([x]) == x` (identity — OR of one)           | — |
| SIG-6 | `make_signature([A, B]) == A \| B` (OR-reduce)               | — |
| SIG-7 | `signifies(0, anything) == False` (vacuous for 0)            | — |
| SIG-9 | `signifies(T(0b110), T(0b010)) == True` (overlapping type bits) | — |
| SIG-10 | `signifies(T(0b100), T(0b010)) == False` (no overlapping type bits) | — |
| SIG-14 | `make_signature([0b10, 0b100]) == 0b110` (OR-reduction of two distinct node values) | — |
| SIG-15 | `signifies(0b110, 0b010) == False` (lower/BPE bits masked off) | — |
| SIG-16 | `signifies(T(0b110) \| 1, T(0b010) \| 2) == True` (type overlap beats differing BPE ids) | — |
| SIG-17 | `residual(T(0b110), T(0b010)) == T(0b100)` (type-word bits in a not in b) | — |
| SIG-18 | `residual(a, a) == 0` for any a (empty residual) | — |
| SIG-19 | `residual(T(0b110) \| 5, T(0b010) \| 7) == T(0b100)` (BPE-id bits masked off) | — |
| SIG-20 | `classify_misfit(T(0b110), [T(0b010)]) == (True, False)` (signature over-claims) | — |
| SIG-21 | `classify_misfit(T(0b010), [T(0b110)]) == (False, True)` (nodes over-deliver) | — |
| SIG-22 | `classify_misfit(T(0b110), [T(0b110)]) == (False, False)` (faithful coverage) | — |
| SIG-23 | `classify_misfit(T(0b100) \| 5, [T(0b100) \| 9]) == (False, False)` (BPE-id difference ignored) | — |

## What a Signifier is Not

The following are explicitly **out of scope** for this spec:

- **Node construction.** How text becomes nodes is defined in the
  @tokenizer spec. The Signifier consumes nodes; it never touches text.
- **Node unpacking.** Extracting the BPE token id from a node
  (`& 0xFFFFFFFF`) is a Tokenizer-layout concern, to be addressed by a
  Tokenizer accessor — not a Signifier operation.
- **Significance computation.** The distance↔significance inversion
  (`~distance & MASK64`) operates on significance magnitude, not on the
  node/signature algebra. It stays in the @model spec / `expand` and is
  not a Signifier concern.
- **The signature concept.** What a signature *is* (its role as a kline
  head value) is defined in the @signature spec. The Signifier defines
  only how signatures are produced and compared.
- **Kline structural predicates.** `is_identity` and `is_canon` are
  structural predicates over a kline's signature and nodes, defined in the
  @kline spec. (`is_canon` delegates reduction to the Signifier but is not
  itself a Signifier operation.)

## Referenced By

- **Signature** (@signature spec) — delegates creation and matching to the
  Signifier.
- **Tokenizer** (@tokenizer spec) — the Signifier consumes the Tokenizer's
  node values (data flow); the NLP pair additionally share a packing
  agreement (see §NLPSignifier).
- **Model** (@model spec) — candidate retrieval uses `signifies`.
- **Agent** (@agent spec) — prepares signatures via `make_signature`.
- **STM** (@stm spec) — computes nodes signatures via `make_signature`.
- **Cogitator** (@cogitator spec) — uses `make_signature` during S2 misfit
  classification.
- **NLP Tokenizer** (@nlp_tokenizer spec) — the NLP deployment bundles the
  NLP Tokenizer with `NLPSignifier`.
