# Tokenizer Specification

## Overview

The tokenizer converts between text and nodes. It is the sole authority
for how text becomes nodes.

Kalvin ships a single production tokenizer. It is built on a **BPE**
subword base and combines each BPE token with a **type word** to form a
64-bit typed node:

- **BPE** — byte-pair encoding. Vocabulary learned from a training corpus.
  Tokens are sequential vocabulary indices.
- **Type word** — a 32-bit bit pattern packed into the upper half of each
  node. Kalvin treats the type word as **opaque**: it participates in
  signature construction and matching but kalvin does not interpret what
  the bits mean. The meaning of the type word is a deployment concern; see
  the @nlp_tokenizer spec for the NLP interpretation used by the shipped
  data.

```
node = (type_word << 32) | bpe_token_id
```

The high 32 bits carry the type word; the low 32 bits carry the BPE token
ID. Both halves participate in the same bitwise OR/AND algebra as any
other node.

### Significance and Tokenizers

A tokenizer does **not** encode knowledge; it encodes **dimensionality**.
Every node is a `uint64`, and the system operates on the bit pattern of a
node — not on any meaning assigned to those bits.

- **Signatures** are built from nodes by bitwise OR via `make_signature`
  (defined in the @signature spec); candidate retrieval uses bitwise AND
  (defined in the @signature / @model specs). No masking or special-casing
  is applied — every node contributes its full value.
- **Significance** is computed at the bit level and is **tokenizer-agnostic**.
  Routing and distance operate on node membership and bit overlap, not on
  what the bits mean (see the @significance / @model specs).

The typed-node format conforms to this algebra directly.

## Dependencies

### Kline (@kline spec)

- **node**: a 64-bit unsigned integer. The tokenizer produces nodes.

The tokenizer does not interpret nodes beyond its own encoding.

### Signature (@signature spec)

- Signature creation (`make_signature`) is a plain OR-reduce of node values.
- The tokenizer does not create signatures itself. See the @signature spec.

## Interface

The tokenizer implements:

### `vocab_size → int ≥ 0`

The number of distinct tokens the BPE vocabulary defines.

### `encode(text: str, pad_ws: bool = False) → list[node]`

Convert a string to an ordered sequence of typed nodes.

- Empty string → empty list.
- Each BPE token is combined with its type word:
  `(type_word << 32) | bpe_token_id`.
- Tokens absent from the type dictionary receive the fallback type word
  (`UNKNOWN_TYPE`).
- Each node carries enough information to reconstruct the original text
  via `decode`.

### `encode_bpe(text: str, pad_ws: bool = False) → list[int]`

Low-level accessor: encode text to raw BPE token IDs (no type word).
Used by training/tagging tooling that operates on the BPE vocabulary
directly.

### `decode(nodes: list[node]) → str`

Convert a sequence of nodes back to a string.

- Empty list → empty string.
- The BPE token ID is taken from the low 32 bits of each node, so raw
  BPE IDs (which fit in 32 bits) decode unchanged.
- `decode(encode(text)) == text` for any string the tokenizer can represent.

## Type Dictionary

The tokenizer owns a **type dictionary** mapping BPE token IDs to their
type word. Entries are loaded from a tagged-grammar file
(`{tokenizer_name}_tagged_grammar.json`) in which every BPE token —
including sub-words — carries a `type_word` field. Other fields in an
entry are opaque metadata, preserved unchanged (the shipped dictionary
carries NLP labels; see the @nlp_tokenizer spec).

BPE tokens without a dictionary entry fall back to `UNKNOWN_TYPE` (the
empty type word, `0`). A deployment may reinterpret this via a subclass
(the NLP specialisation uses `POS_X`).

### Lookup

- `lookup_type(token_id) → int | None` — the type word for a BPE token ID.
- `lookup_type_entry(token_id) → dict | None` — the raw dictionary entry.

## BPE Engine

### Vocabulary

- Learned from a text corpus via training.
- Default target size: 32,768 (actual trained vocab may be smaller).

### Training

```
tokenizer.train(texts: Iterator[str], vocab_size: int, pattern: str | None) → void
```

- `texts` — iterator over training strings.
- `vocab_size` — target vocabulary size (minimum 256).
- `pattern` — optional regex pattern for pre-tokenization.

Training is required before `encode`/`decode` can be used. Training loads
the BPE engine only; the type dictionary is loaded separately.

### Persistence

```
tokenizer.save_to_directory(path, name) → void
tokenizer.from_directory(path, name) → Tokenizer
```

`from_directory` loads the BPE engine only (no type dictionary); `encode`
on the result produces nodes with the fallback type word for every token.
Use `from_files` to load both the engine and a type dictionary.

### Production Factory

```
tokenizer.from_files(tokenizer_path, tokenizer_name) → Tokenizer
```

Loads the BPE engine and type dictionary from standard paths. When
`tokenizer_path` is omitted it is resolved via
`kalvin.paths.tokenizer_dir()`.

### Encoding

BPE segments text into subword units, each mapped to a vocabulary index.
Each BPE token ID is then combined with its type word from the type
dictionary:

```
encode("hello") → [(type_word("hello") << 32) | 15496]
```

#### Worked Example

```
BPE encode("the air") → [257, 500]

Type dictionary lookup:
  257 "the" → type_word = T_the
  500 "air" → type_word = T_air

Typed nodes: [(T_the << 32) | 257, (T_air << 32) | 500]
```

#### Signature construction

```
nodes = [(T_the << 32) | 257, (T_air << 32) | 500]

make_signature(nodes) → nodes[0] | nodes[1]
```

The signature captures both the type words and the BPE token IDs.

> **Note.** `make_signature` is defined in the @signature spec, not here.

## Signature Behavior

`make_signature()` OR-reduces the full unmasked node values. Both the
type word (high 32) and the BPE token ID (low 32) contribute to the
signature. The tokenizer passes raw typed nodes to `make_signature()` as
it would any other node sequence.

## Test Matrix

| ID    | Criterion                                                                  | Origin ref |
| ----- | -------------------------------------------------------------------------- | ---------- |
| TOK-7 | Empty string: `encode("") == []`, `decode([]) == ""`                        | — |
| TOK-8 | Typed-node format: every node is `(type_word << 32) \| bpe_token_id`        | — |
| TOK-9 | Round-trip: `decode(encode(text)) == text` for representable text           | — |
| TOK-10 | Type dictionary: `lookup_type(id)` returns the entry's `type_word`         | — |
| TOK-11 | Unknown-token fallback: tokens absent from the dictionary get `UNKNOWN_TYPE` | — |
| TOK-12 | `from_files()` loads the BPE engine and type dictionary                    | — |

NLP-specific criteria (NLP type-word interpretation, dimension count,
curriculum compatibility) live in the @nlp_tokenizer spec.

## What a Tokenizer is Not

The following are explicitly **out of scope** for this spec:

- **Signature creation.** Signature construction (`make_signature`) is
  defined in the @signature spec.
- **Significance computation.** Significance is defined in the
  @significance spec. The tokenizer does not compute or store significance.
- **Interpretation of the type word.** What the type-word bits mean is a
  deployment concern. The NLP interpretation is specified in the
  @nlp_tokenizer spec; the base tokenizer treats the type word as opaque.
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
- **NLP Tokenizer** (@nlp_tokenizer spec) — the NLP interpretation of the
  type word used by the shipped data.
