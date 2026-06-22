# Tokenizer Specification

## Overview

The tokenizer converts between text and nodes. It is the sole authority
for how text becomes nodes.

The tokenizer *interface* (`KTokenizer`) is layout-agnostic: it converts
text to an ordered sequence of uint64 nodes and back. It does not specify
how nodes are packed, nor anything about type words. Bit-packing, type
dictionaries, and node layouts are concrete-tokenizer concerns — see the
@nlp_tokenizer spec for the production tokenizer, which packs a type word
into the upper 32 bits of each node and a BPE token ID into the lower 32.

Kalvin ships a single production tokenizer (`NLPTokenizer`), built on a
**BPE** subword base. A BPE subword vocabulary is a sequence of vocabulary
indices learned from a training corpus.

```
node = (type_word << 32) | bpe_token_id      # the NLP production layout
```

The high 32 bits carry the type word; the low 32 bits carry the BPE token
ID. This layout is **a property of the NLP production tokenizer**,
documented here for context and defined authoritatively in @nlp_tokenizer.

### Significance and Tokenizers

A tokenizer does **not** encode knowledge; it encodes **dimensionality**.
Every node is a `uint64`, and the system operates on the node's value —
not on any meaning assigned to its bits.

- **Signatures** are built from nodes and compared for candidate retrieval
  via the operations defined in the @signifier spec. The tokenizer does
  not define these operations.
- **Significance** is **tokenizer-agnostic**. Routing and distance operate
  on node membership and bit overlap, not on what the bits mean (see the
  @significance / @model specs).

## Dependencies

### Kline (@kline spec)

- **node**: a 64-bit unsigned integer. The tokenizer produces nodes.

The tokenizer does not interpret nodes beyond its own encoding.

### Signature (@signature spec)

- A signature is the uint64 value occupying a kline's head position.

### Signifier (@signifier spec)

- Signature creation (`make_signature`) and overlap matching (`signifies`)
  are defined in the @signifier spec. The Signifier consumes the nodes the
  Tokenizer produces; the tokenizer does not create signatures itself.

## Interface (`KTokenizer`)

The interface specifies role only — text↔nodes. It does not specify a
node layout, a type-word concept, or any bit packing; those are
concrete-tokenizer concerns (see @nlp_tokenizer).

### `vocab_size → int ≥ 0`

The size of the tokenizer's vocabulary.

### `encode(text: str, pad_ws: bool = False) → list[node]`

Convert a string to an ordered sequence of nodes.

- Empty string → empty list.
- Each node carries enough information to reconstruct the original text
  via `decode`.

### `decode(nodes: list[node]) → str`

Convert a sequence of nodes back to a string.

- Empty list → empty string.
- `decode(encode(text)) == text` for any string the tokenizer can represent.

## BPE Engine (foundation of the production tokenizer)

The production tokenizer is built on a BPE subword vocabulary. The BPE
engine is the *foundation* of the NLP production tokenizer (see
@nlp_tokenizer): it manages the subword vocabulary and exposes raw
BPE↔text operations. It is not a `KTokenizer` on its own — a BPE engine
produces raw vocabulary indices, not nodes.

### Vocabulary

- Learned from a text corpus via training.
- Default target size: 32,768 (actual trained vocab may be smaller).

### Training

```
engine.train(texts: Iterator[str], vocab_size: int, pattern: str | None) → void
```

- `texts` — iterator over training strings.
- `vocab_size` — target vocabulary size (minimum 256).
- `pattern` — optional regex pattern for pre-tokenization.

Training loads the BPE engine only; the type dictionary (if any) is a
concrete-tokenizer concern loaded separately (see @nlp_tokenizer).

### Persistence

```
engine.save_to_directory(path, name) → void
engine.load_from_directory(path, name) → engine
```

These manage the BPE vocabulary only. Building a `KTokenizer` from them is
a concrete-tokenizer concern (see @nlp_tokenizer, Construction).

## Signature Behavior

A concrete tokenizer passes its nodes to `make_signature()` as it would
any other node sequence; the reduction is defined in the @signifier spec.

## Test Matrix

The interface has no layout-specific criteria — its contract is role only.
Layout and type criteria live in @nlp_tokenizer.

| ID    | Criterion                                                                  | Origin ref |
| ----- | -------------------------------------------------------------------------- | ---------- |
| TOK-7 | Empty string: `encode("") == []`, `decode([]) == ""`                        | — |
| TOK-9 | Round-trip: `decode(encode(text)) == text` for representable text           | — |

## What a Tokenizer is Not

The following are explicitly **out of scope** for this spec:

- **Node layout / packing.** How a concrete tokenizer packs text
  information into a 64-bit node (type words, BPE ids, masks) is a
  concrete-tokenizer concern. The production layout is defined in
  @nlp_tokenizer.
- **Type dictionaries.** Type words, type lookup, and unknown-token
  fallback are concrete-tokenizer concerns (see @nlp_tokenizer).
- **Signature creation.** Signature construction (`make_signature`) is
  defined in the @signifier spec.
- **Significance computation.** Significance is defined in the
  @significance spec. The tokenizer does not compute or store significance.
- **Model operations.** Storing, retrieving, and querying klines are model
  concerns.
- **Training data format.** How training data is sourced and preprocessed
  for BPE training is an implementation concern.

## Referenced By

- **Signature** (@signature spec) — signature creation consumes nodes
  produced by the tokenizer.
- **Kline** (@kline spec) — klines contain nodes produced by the tokenizer.
- **Agent** (@agent spec) — uses the tokenizer to encode input and
  construct klines.
- **Significance** (@significance spec) — consumes nodes produced by the
  tokenizer (nodes are opaque to significance).
- **NLP Tokenizer** (@nlp_tokenizer spec) — the production tokenizer;
  owns the node layout, type dictionary, and the BPE-engine foundation.
