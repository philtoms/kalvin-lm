# Tokenizer Specification

## Overview

The tokenizer converts between text and nodes. It is the sole authority
for how text becomes nodes.

Two tokenizer types are defined, both conforming to the same interface:

- **BPE** — byte-pair encoding. Vocabulary learned from a training corpus.
  Tokens are sequential vocabulary indices, combined with type prefixes to
  form typed nodes.
- **Mod** — modular bit-packed encoding. Vocabulary is a fixed character set.
  Tokens are bit positions with bitwise OR/AND semantics.

Both types ultimately produce the same kind of output: typed nodes
suitable for signature construction (defined in the @signature spec).

## Dependencies

### Kline (@kline spec)

- **node**: a 64-bit unsigned integer. The tokenizer produces nodes.

The tokenizer does not interpret nodes beyond its own encoding.

### Signature (@signature spec)

- Signature creation (`make_signature`) depends on the tokenizer's
  `is_literal` function to determine which nodes contribute.
- The tokenizer does not create signatures itself. See the @signature spec.

## Interface

All tokenizer types implement:

### `vocab_size → int ≥ 0`

The number of distinct tokens the vocabulary defines.

### `encode(text: str) → list[node]`

Convert a string to an ordered sequence of nodes.

- Empty string → empty list.
- Each node carries enough information to reconstruct the original text
  via `decode`.

### `decode(nodes: list[node]) → str`

Convert a sequence of nodes back to a string.

- Empty list → empty string.
- `decode(encode(text)) == text` for any string the tokenizer can represent.

### `is_literal(node: int) → bool`

Returns whether the node represents a literal token.

- Literal tokens preserve character identity and order.
- Non-literal (typed) tokens carry structural or type information and
  participate in bitwise OR signature construction.

## Vocabulary

A vocabulary is the ordered set of symbols the tokenizer can encode.

Both tokenizer types provide a default vocabulary. The default may be
overridden at initialisation.

| Type | Default                                               |
| ---- | ----------------------------------------------------- |
| BPE  | 4096 entries (learned from corpus)                    |
| Mod  | 95 printable ASCII characters (codes 32–126), ordered |

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

### Literal Test

BPE tokens are never literal:

```
is_literal(node) → False   (always)
```

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

## Mod Tokenizer

### Overview

A Mod tokenizer maps characters to bit positions within a node. Strings
are encoded as the bitwise OR of constituent character bits, producing a
single node. Individual characters can also be encoded as literal tokens
that preserve order and identity.

Two encoding modes produce two token categories:

| Mode    | Tokens   | Bit 0 | Order preserved | Use                   |
| ------- | -------- | ----- | --------------- | --------------------- |
| Packed  | Single   | 0     | No              | Signatures, AND match |
| Literal | Per-char | 1     | Yes             | Exact text, sequence  |

### Bit Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ Bit 0       │ LITERAL flag: 0 = packed, 1 = literal              │
│ Bits 1–N    │ Character bits (N determined by variant)           │
│ Bits N+1–63 │ Unused                                             │
└──────────────────────────────────────────────────────────────────┘
```

### Variants

| Variant | Character bits | Bit range | Fits uint64 |
| ------- | -------------- | --------- | ----------- |
| Mod32   | 31             | Bits 1–31 | Yes         |
| Mod64   | 63             | Bits 1–63 | Yes         |

### Vocabulary

The vocabulary is an ordered string of characters. Each character maps to
a single bit position (bit 1, bit 2, …, bit N). When the character count
exceeds N, positions wrap.

Default vocabulary:

```
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \"',.;:!?/\n\t%{}[]()<>#$@£^&*+-_=
```

All printable ASCII characters (codes 32–126) must be mappable.
Characters not in the explicit vocabulary are assigned the next available
bit position (wrapping).

### Packed Encoding

Multi-character strings are OR-ed into a single node. Bit 0 is clear.

```
encode("ABC") → [CHAR_BIT['A'] | CHAR_BIT['B'] | CHAR_BIT['C']]
```

Properties:

- Exactly one node per string.
- Order is lost: `"AB"` and `"BA"` produce the same node.
- Multiplicity is lost: `"AA"` and `"A"` produce the same node.
- Suitable for signature construction and bitwise AND matching.

### Literal Encoding

Each character becomes a separate node. The upper 32 bits store the
Unicode code point. The lower 32 bits are all set (`0xFFFFFFFF`) as a
**literal mask** — a bit pattern that uniquely identifies the node as
literal and distinguishes it from packed nodes and signature values.

```
encode("ABC", literal=True) → [(65 << 32) | 0xFFFFFFFF,
                                (66 << 32) | 0xFFFFFFFF,
                                (67 << 32) | 0xFFFFFFFF]
```

Properties:

- One node per character.
- Order is preserved.
- Identity is preserved (distinct characters → distinct tokens).
- Bypasses the vocabulary: any character with a valid code point can be
  encoded, including characters not in the vocabulary.
- The literal mask (`0xFFFFFFFF`) occupies the lower 32 bits, keeping the
  character code point in the upper 32 bits. This avoids collision with
  packed nodes (which use bits 1–31 or 1–63) and with signatures (where
  bit 0 is the literal-content flag, not a full 32-bit mask).
- A raw integer may also be encoded: `encode(42, literal=True)`
  → `[(42 << 32) | 0xFFFFFFFF]`.

### Decoding

Auto-detects encoding mode from the literal mask:

```
if (node & 0xFFFFFFFF) == 0xFFFFFFFF  →  literal: chr(node >> 32)
else                                   →  packed: find all set bits, map each to character
```

Packed decode returns characters in bit-position order (lowest bit first),
not in the original text order.

### Literal Test

```
is_literal(node) → (node & 0xFFFFFFFF) == 0xFFFFFFFF
```

The lower 32 bits being all set is the literal mask. This distinguishes
literal tokens from packed nodes (which have bit 0 clear) and from
signature values (where bit 0 may be set but the lower 32 bits are not all
1s).

### Worked Examples

#### Packed encoding (Mod32)

```
Vocabulary: "ABC" → bit 1, bit 2, bit 3

encode("A")       → [0b10]                  = [2]
encode("B")       → [0b100]                 = [4]
encode("AB")      → [0b10 | 0b100]          = [6]
encode("ABC")     → [0b10 | 0b100 | 0b1000] = [14]
```

#### Literal encoding (Mod32)

```
encode("AB", literal=True) → [(65 << 32) | 0xFFFFFFFF, (66 << 32) | 0xFFFFFFFF]
                            → [4294967361, 4294967362]

decode([4294967361, 4294967362]) → "AB"

is_literal(4294967361) → True
is_literal(2)   → False
```

#### Bitwise properties of packed nodes

Packed (non-literal) nodes can be combined with bitwise OR and tested
with bitwise AND. These properties are used by signature construction
(@signature spec) and candidate retrieval (@model spec):

```
2 | 4 = 6              # 'A' | 'B' = combined
(6 & 2) != 0 → True   # combined contains 'A'
```

## What a Tokenizer is Not

The following are explicitly **out of scope** for this spec:

- **Signature creation.** Signature construction (`make_signature`) is
  defined in the @signature spec. The tokenizer provides `is_literal` but
  does not produce signatures.
- **Significance computation.** Significance is defined in the
  @significance spec. The tokenizer does not compute or store significance.
- **Type prefix assignment.** How linguistic types are assigned to BPE
  tokens is an agent concern. The tokenizer provides raw token IDs.
- **Model operations.** Storing, retrieving, and querying klines are model
  concerns.
- **Training data format.** How training data is sourced and preprocessed
  for BPE training is an implementation concern.

## Referenced By

- **Signature** (@signature spec) — signature creation depends on the
  tokenizer's `is_literal` function.
- **Kline** (@kline spec) — klines contain nodes produced by the tokenizer.
- **Agent** (@agent spec) — uses the tokenizer to encode input, apply type
  prefixes, and construct klines.
- **Significance** (@significance spec) — consumes nodes produced by the
  tokenizer (nodes are opaque to significance).
