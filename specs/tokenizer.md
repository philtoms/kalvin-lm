# Tokenizer Specification

## Overview

The tokenizer converts between text and nodes. It is the sole authority
for how text becomes nodes.

The system ships a single production tokenizer, **NLP**, built on a **BPE**
subword base:

- **BPE** — byte-pair encoding. Vocabulary learned from a training corpus.
  Tokens are sequential vocabulary indices, combined with type prefixes to
  form typed nodes. BPE is the subword foundation that NLP extends.
- **NLP** — hybrid BPE + NLP type encoding (the production default). BPE
  subword tokens are combined with NLP type information (POS + DEP + MORPH)
  in a single 64-bit node. The tokenizer owns the grammar dictionary that
  maps BPE tokens to NLP types.

Both produce the same kind of output: typed nodes suitable for signature
construction (defined in the @signature spec).

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

The NLP node format conforms to this algebra directly:

```
node = (nlp_type32 << 32) | bpe_token_id
```

The high 32 bits carry NLP type dimensions (POS + DEP + MORPH); the low 32
bits carry the BPE token ID. Both halves participate in the same bitwise
OR/AND algebra as any other node.

## Dependencies

### Kline (@kline spec)

- **node**: a 64-bit unsigned integer. The tokenizer produces nodes.

The tokenizer does not interpret nodes beyond its own encoding.

### Signature (@signature spec)

- Signature creation (`make_signature`) is a plain OR-reduce of node values.
- The tokenizer does not create signatures itself. See the @signature spec.

## Interface

All tokenizer types implement:

### `vocab_size → int ≥ 0`

The number of distinct tokens the vocabulary defines.

### `encode(text: str) → list[node]`

Convert a string to an ordered sequence of nodes.

- Empty string → empty list.
- The encoding mode is determined by the tokenizer type and input content.
- Each node carries enough information to reconstruct the original text
  via `decode`.

### `decode(nodes: list[node]) → str`

Convert a sequence of nodes back to a string.

- Empty list → empty string.
- `decode(encode(text)) == text` for any string the tokenizer can represent.

## Vocabulary

A vocabulary is the ordered set of symbols the tokenizer can encode.

Both tokenizer types provide a default vocabulary. The default may be
overridden at initialisation.

| Type | Default                                               |
| ---- | ----------------------------------------------------- |
| BPE  | 4096 entries (learned from corpus)                    |
| NLP  | 17,392 BPE entries + 12,871 grammar dictionary entries |

## BPE Tokenizer

### Overview

A BPE tokenizer learns subword token vocabulary from a training corpus.
Tokens are sequential integer IDs assigned during training.

BPE is suitable for general-purpose text encoding where subword
segmentation and large vocabularies are needed.

### Vocabulary

- Learned from a text corpus via training.
- Default target size: 4096.
- Configurable at training time.

### Training

```
tokenizer.train(texts: Iterator[str], vocab_size: int, pattern: str | None) → void
```

- `texts` — iterator over training strings.
- `vocab_size` — target vocabulary size (minimum 256).
- `pattern` — optional regex pattern for pre-tokenization.

Training is required before `encode`/`decode` can be used.

### Persistence

```
tokenizer.save_to_directory(path, name) → void
tokenizer.from_directory(path, name) → Tokenizer
```

Save and load trained tokenizer state. Format is implementation-defined.

### Type Prefixes

BPE token IDs are sequential integers without inherent type information.
To produce typed nodes suitable for signature construction, each BPE token
is combined with a **type prefix** — a bit pattern encoding linguistic
properties (part-of-speech, dependency, morphology, etc.).

Type prefixes are external to the tokenizer. They are provided by the
agent layer via a dictionary lookup at encode time:

```
for token in bpe_tokens:
    type_prefix = dictionary.lookup(token)   # linguistic type bits
    node = type_prefix | token               # typed node
```

Unrecognised tokens receive a default type prefix (e.g. `POS_X`).
Whitespace tokens may pass through as-is (no type prefix).

### Encoding

BPE segments text into subword units, each mapped to a vocabulary index:

```
encode("hello") → [15496]
encode("hello world") → [15496, 995]
```

The returned values are raw BPE token IDs. Type prefix combination
happens at the agent layer.

### Worked Examples

#### Encoding with type prefixes

```
BPE encode("the air") → [257, 500]

Agent applies type prefixes from dictionary:
  257 "the" → POS_DET | 257 = 4194337
  500 "air" → POS_NOUN | DEP_OBJ | 500 = 262788

Typed nodes: [4194337, 262788]
```

#### Signature construction

```
nodes = [4194337, 262788]

make_signature(nodes) → 4194337 | 262788 = 4456397

Signature captures: token IDs 257 and 500 are present,
with POS_DET and POS_NOUN | DEP_OBJ type information.
```

> **Note.** `make_signature` is defined in the @signature spec, not here.
> This example illustrates how typed BPE nodes feed into it.

## NLP Tokenizer

### Overview

An NLP tokenizer is a hybrid that combines BPE subword encoding with NLP
type information (POS + DEP + MORPH) in a single 64-bit node. Every word
becomes one or more NLP-BPE tokens.

If a word BPE-encodes into multiple subword tokens (e.g. "unhappiness" →
`[un, ##happiness]`), each subword token gets its own grammar lookup.
Subwords without dictionary entries fall back to `POS_X`.

NLP is suitable for text encoding where linguistic type information must
be embedded directly in each node, enabling signature-based similarity
matching that reflects grammatical structure.

### Node Format

Each NLP-BPE node is a 64-bit unsigned integer:

```
node = (nlp_type32 << 32) | bpe_token_id
```

Bit layout:

```
┌──────────────────────────────────────────────────────────────────┐
│ High 32 bits (32–63) │ nlp_type32 — NLP type encoding            │
│                      │   Bits 0–16:  17 POS tags                  │
│                      │   Bits 17–24: 8 DEP groups                 │
│                      │   Bits 25–31: 7 MORPH features             │
│ Low 32 bits (0–31)  │ BPE token ID (0–17391)                     │
└──────────────────────────────────────────────────────────────────┘
```

The `nlp_type32` value is a 32-bit bitmask encoding 32 linguistic
dimensions (17 POS + 8 DEP + 7 MORPH = 32 flags), sourced from the
32-bit NLP type legend. The BPE token ID occupies the low 32 bits.

### Fallback for Unknown BPE Tokens

BPE tokens without a grammar dictionary entry receive `POS_X = 65536
= 1 << 16` as their `nlp_type32`. This ensures unknown words still have
a valid NLP type, preserving the node format invariant.

### Vocabulary

- **BPE vocabulary**: 17,392 tokens (from `tokenizer-32768.json`, where
  32,768 was the training target; actual trained vocab is 17,392).
- **Grammar dictionary**: 12,871 BPE→NLP entries mapping BPE token IDs
  to `nlp_type32` values (from `simplestories-1_grammar.json`).

Source files:

| File | Purpose |
| ---- | ------- |
| `data/tokenizer/tokenizer-32768.json` | BPE vocabulary |
| `data/tokenizer/simplestories-1_grammar.json` | Grammar dictionary (BPE→NLP) |
| `data/tokenizer/simplestories-1_nlp_type32.json` | 32-bit NLP type legend |

### Encoding

The encode process:

1. **BPE-encode** the text into subword token IDs.
2. **Grammar lookup** — for each token ID, look up the `nlp_type32` in
   the grammar dictionary. Tokens without entries default to `POS_X`.
3. **Construct nodes** — assemble each node as
   `(nlp_type32 << 32) | bpe_token_id`.

#### Worked Example

```
encode("Tea brewed softly")

Step 1 — BPE encode:
  "Tea"     → token 12465
  "brewed"  → token 4964
  "softly"  → token 977

Step 2 — Grammar lookup (nlp_type32 from dictionary):
  12465 "Tea"     → nlp_type32 = 131200
  4964  "brewed"  → nlp_type32 = 8421376
  977   "softly"  → nlp_type32 = 2097156

Step 3 — Node construction:
  "Tea"     → (131200  << 32) | 12465 = 563499709247665
  "brewed"  → (8421376 << 32) | 4964  = 36169534507324260
  "softly"  → (2097156 << 32) | 977   = 9007216434611153

Result: [563499709247665, 36169534507324260, 9007216434611153]
```

### Decoding

Extract the BPE token ID from the low 32 bits and decode via the BPE
vocabulary:

```
bpe_token_id = node & 0xFFFFFFFF
text = bpe_vocab.decode(bpe_token_id)
```

NLP type bits (high 32) are not needed for text reconstruction.

### Dimension Count

The `nlp_type32` encoding provides **32 dimensions**:

- 17 POS tags (bits 0–16)
- 8 DEP groups (bits 17–24)
- 7 MORPH features (bits 25–31)

Comparison with the design target:

| Tokenizer | Dimensions | Notes |
| --------- | ---------- | ----- |
| NLP       | 32         | 17 POS + 8 DEP + 7 MORPH |
| Target    | ~35        | From CONTEXT.md |

### Signature Behavior

`make_signature()` for NLP nodes OR-reduces the full unmasked node values.
Both NLP type bits (high 32) and BPE token IDs (low 32) contribute to
the signature. The tokenizer passes raw NLP-BPE nodes to `make_signature()`
as it would any other node sequence.

### Worked Examples

#### Encoding with grammar lookup

```
NLP encode("Tea brewed softly")

BPE: [12465, 4964, 977]

Grammar dictionary lookups:
  12465 → nlp_type32 = 131200   (POS_PROPN | DEP_SUBJ | MORPH_SING)
  4964  → nlp_type32 = 8421376  (POS_VERB  | DEP_ROOT | MORPH_PAST | MORPH_SING)
  977   → nlp_type32 = 2097156  (POS_ADV   | DEP_ADVMOD)

Nodes: [563499709247665, 36169534507324260, 9007216434611153]
```

#### Round-trip

```
encode("Tea brewed softly") → [563499709247665, 36169534507324260, 9007216434611153]
decode([563499709247665, 36169534507324260, 9007216434611153]) → "Tea brewed softly"
```

#### Signature construction

```
Nodes: [563499709247665, 36169534507324260, 9007216434611153]

make_signature(nodes) →
  563499709247665 | 36169534507324260 | 9007216434611153

Signature captures: both NLP type and BPE ID dimensions — the full
identity of each token.
```

> **Note.** `make_signature` is defined in the @signature spec, not here.

## Test Matrix

| ID    | Criterion                                                                  | Origin ref |
| ----- | -------------------------------------------------------------------------- | ---------- |
| TOK-7 | Empty string: `encode("") == []`, `decode([]) == ""`                        | — |
| TOK-NLP-1 | NLP encode produces correct node format: `(nlp_type32 << 32) \| bpe_token_id` for each token | NLP |
| TOK-NLP-2 | NLP encode with unknown BPE token uses `POS_X = 65536` as `nlp_type32`    | NLP |
| TOK-NLP-4 | NLP round-trip: `decode(encode("Tea brewed softly")) == "Tea brewed softly"` | NLP |
| TOK-NLP-7 | Vocabulary sizes: BPE vocab = 17,392 tokens, grammar dictionary = 12,871 entries | NLP |
| TOK-NLP-8 | Dimension count: `nlp_type32` provides 32 dimensions (17 POS + 8 DEP + 7 MORPH) | NLP |

## What a Tokenizer is Not

The following are explicitly **out of scope** for this spec:

- **Signature creation.** Signature construction (`make_signature`) is
  defined in the @signature spec.
- **Significance computation.** Significance is defined in the
  @significance spec. The tokenizer does not compute or store significance.
- **Type prefix assignment (BPE).** How linguistic types are assigned to
  BPE tokens is an agent concern. The BPE tokenizer provides raw token IDs.
  **Note:** For the NLP tokenizer, grammar lookups are intrinsic — the
  tokenizer owns the dictionary. This distinction is by design: BPE type
  prefixes are external (agent layer), while NLP grammar lookups are
  internal (tokenizer layer).
- **Model operations.** Storing, retrieving, and querying klines are model
  concerns.
- **Training data format.** How training data is sourced and preprocessed
  for BPE training is an implementation concern.

## Referenced By

- **Signature** (@signature spec) — signature creation consumes nodes
  produced by the tokenizer.
- **Kline** (@kline spec) — klines contain nodes produced by the tokenizer.
- **Agent** (@agent spec) — uses the tokenizer to encode input, apply type
  prefixes, and construct klines.
- **Significance** (@significance spec) — consumes nodes produced by the
  tokenizer (nodes are opaque to significance).
