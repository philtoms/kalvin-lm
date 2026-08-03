# Kline Specification

## Overview

A Kline is the fundamental unit of Kalvin's memory. It is an identified,
ordered sequence of zero or more nodes.

## Definition

A Kline consists of:

| Field     | Type               | Description                        |
| --------- | ------------------ | ---------------------------------- |
| signature | uint64             | Identity key.                      |
| nodes     | sequence of uint64 | Zero or more nodes. Ordered. |

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

A kline's structural kind is determined by its signature and nodes (no model
state). The kinds are defined in @CONTEXT.md §Structure: **Terminal** is the
genus of **Unknown** and **Identity**; **Canon** and **Misfit** are the
non-terminal structures. The predicates below expose these kinds to
rationalisation.

A §11.3 **compound-word** — a single word the external tokenizer
split into BPE subwords — carries the boundary marker token
`COMPOUND_TOKEN` (@nlp_tokenizer spec) as an extra node: `Mary: [COMPOUND_TOKEN, M, ary]`.
The token participates in the signature algebra like any other node, so the
compound's signature _encodes_ the marker naturally
(`signature == signature_of([M, ary, COMPOUND_TOKEN])`) — no bit masking
anywhere.

- **`is_compound_word(kline)`** — `True` iff `COMPOUND_TOKEN` is among the
  kline's nodes. The compiler appends the token only to a compound-word's
  nodes. The marker is confined to the kalvin↔NLP boundary: defined in
  `nlp_tokenizer.py`, appended by `ks/token_encoder.py`, read here; no other
  module names it, and the signifier treats it as an ordinary node (no
  masking).
- **`is_terminal(kline)`** — `True` for any kline that carries no further
  decomposition: the empty form `{S: []}`, the self-referential form
  `{S: [S]}` (sole node equals signature), or a compound-word
  (`is_compound_word`). A terminal is a leaf that tells traversal to stop
  (see @CONTEXT.md §Terminal). It is the genus of `is_unknown` and
  `is_identity`.
- **`is_unknown(kline)`** — `True` for the empty form `{S: []}` only. An
  Unknown claims **S4** — nothing held for this signature; the structural
  form of an ask (see @CONTEXT.md §Unknown).
- **`is_identity(kline)`** — `True` for a terminal that is directly decodable:
  the self-referential form `{S: [S]}` (a value that decodes into itself), or
  a compound-word (the word is one lexical item; its subwords are an
  encoding artefact). An Identity claims **S1**. Both forms overrule any
  canon classification (see `is_canon` and @CONTEXT.md §Identity).
- **`is_canon(kline)`** — `True` when the kline is _not_ terminal AND
  `signature == signature_of(nodes)`. Canons are non-terminal: a terminal
  is never a canon.
- **`is_misfit(kline)`** — `True` for a non-terminal kline whose
  `signature != signature_of(nodes)` (the residual case after terminal and
  canon are excluded). A Misfit claims **S2** (@CONTEXT.md §Misfit); the
  underfit/overfit/dual residuals are classified by the @signifier spec
  (`classify_misfit`).

These live with the KLine because they are structural properties; the model
and significance modules consume them.

| ID    | Criterion                                                                        |
| ----- | -------------------------------------------------------------------------------- |
| KL-20 | `is_unknown({S: []})` → True (empty form is Unknown, not Identity)               |
| KL-20a | `is_terminal({S: []})` → True                                                   |
| KL-20b | `is_identity({S: []})` → False (empty form is Unknown)                         |
| KL-21 | `is_identity({S: [S]})` → True (self-referential, decodable)                    |
| KL-21a | `is_terminal({S: [S]})` → True                                                  |
| KL-22 | `is_identity({S: [A]})` (A ≠ S) → False                                          |
| KL-23 | `is_canon({S: [A, B]})` where `S == A\|B` and `S` not in nodes → True          |
| KL-24 | `is_canon({S: [S]})` → False (terminal, not canon)                              |
| KL-25 | `is_canon({S: []})` → False (terminal, not canon)                               |
| KL-26 | `is_identity({S: [COMPOUND_TOKEN, A, B]})` → True (compound-word, decodable)    |
| KL-26a | `is_terminal({S: [COMPOUND_TOKEN, A, B]})` → True                               |
| KL-27 | `is_canon({S: [COMPOUND_TOKEN, A, B]})` → False (terminal, not canon)            |
| KL-28 | `is_misfit({AB: [C, D]})` (signature ≠ signature_of(nodes)) → True              |
| KL-29 | `is_misfit({S: [A, B]})` where `S == A\|B` (canon) → False                     |
| KL-30 | `is_misfit({S: []})` → False (terminal, not misfit)                              |
| KL-31 | `is_misfit({S: [S]})` → False (terminal, not misfit)                             |

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

- **Significance** (@model spec §Significance Semantics) — compares query
  and candidate Klines.
- **Model** (@model spec) — stores and retrieves Klines by signature.
- **Rationaliser** (@agent spec) — encodes input into Klines, retrieves candidates.
