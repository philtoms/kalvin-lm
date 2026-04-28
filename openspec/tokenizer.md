# Tokenizer Specification

## Overview

The tokenizer converts between text and KNodes, and produces KLine signatures
from node sets. It is the sole authority for how text becomes nodes and how
node sets become signatures.

Two tokenizer types are defined, both conforming to the same interface:

- **BPE** — byte-pair encoding. Vocabulary learned from a training corpus.
  Tokens are sequential vocabulary indices, combined with type prefixes to
  form typed nodes.
- **Mod** — modular bit-packed encoding. Vocabulary is a fixed character set.
  Tokens are bit positions with bitwise OR/AND semantics.

Both types ultimately produce the same kind of output: a KLine where the
signature is an OR-reduction of its typed nodes.

## Dependencies

### Kline (@kline spec)
- **KNode**: a 64-bit unsigned integer. The tokenizer produces KNodes.
- **KSig**: a 64-bit unsigned integer. The tokenizer produces KSigs via
  `make_signature`.

The tokenizer does not interpret KNodes or KSigs beyond its own encoding.

## Interface

All tokenizer types implement:

### `vocab_size → int ≥ 0`

The number of distinct tokens the vocabulary defines.

### `encode(text: str) → list[KNode]`

Convert a string to an ordered sequence of KNodes.

- Empty string → empty list.
- Each KNode carries enough information to reconstruct the original text
  via `decode`.

### `decode(nodes: list[KNode]) → str`

Convert a sequence of KNodes back to a string.

- Empty list → empty string.
- `decode(encode(text)) == text` for any string the tokenizer can represent.

### `make_signature(nodes) → KSig`

Produce a kline signature from a node set by OR-reduction of non-literal
nodes.

Properties:

| Property        | Rule                                          |
|-----------------|-----------------------------------------------|
| Deterministic   | Same node set → same signature                |
| Commutative     | Node order does not affect the result         |
| Empty           | `make_signature([]) == 0`                     |
| Identity        | `make_signature([node]) == node` for non-literal node |
| Literal-exclude | Literal nodes do not contribute               |

```
sig = 0
for node in nodes:
    if not is_literal(node):
        sig |= node
return sig
```

> **Design note.** `make_signature` bridges the tokenizer and agent
> domains: the encoding is tokenizer-dependent but the operation
> conceptually belongs to the agent layer, which has knowledge of node
> semantics. This spec defines the contract; implementation placement is
> flexible.

### `is_literal(node: KNode) → bool`

Returns whether the node represents a literal token.

- Literal tokens preserve character identity and order.
- Non-literal (typed) tokens carry structural or type information and
  participate in bitwise OR signature construction.

## Vocabulary

A vocabulary is the ordered set of symbols the tokenizer can encode.

Both tokenizer types provide a default vocabulary. The default may be
overridden at initialisation.

| Type | Default                                              |
|------|------------------------------------------------------|
| BPE  | 4096 entries (learned from corpus)                   |
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

### Signature Construction

`make_signature` performs OR-reduction over non-literal nodes, identical
to the Mod algorithm. Since BPE nodes have already been combined with
type prefixes at encode time, the OR-reduction produces a signature that
captures both token identity and linguistic type:

```
make_signature([POS_NOUN | 500, POS_DET | 257])
    → (POS_NOUN | 500) | (POS_DET | 257)
```

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

## Mod Tokenizer

### Overview

A Mod tokenizer maps characters to bit positions within a KNode. Strings
are encoded as the bitwise OR of constituent character bits, producing a
single node. Individual characters can also be encoded as literal tokens
that preserve order and identity.

Two encoding modes produce two token categories:

| Mode     | Tokens    | Bit 0 | Order preserved | Use                    |
|----------|-----------|-------|-----------------|------------------------|
| Packed   | Single    | 0     | No              | Signatures, AND match  |
| Literal  | Per-char  | 1     | Yes             | Exact text, sequence   |

### Bit Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ Bit 0       │ LITERAL flag: 0 = packed, 1 = literal              │
│ Bits 1–N    │ Character bits (N determined by variant)           │
│ Bits N+1–63 │ Unused                                             │
└──────────────────────────────────────────────────────────────────┘
```

### Variants

| Variant | Character bits | Bit range  | Fits uint64 |
|---------|---------------|------------|-------------|
| Mod32   | 31            | Bits 1–31  | Yes         |
| Mod64   | 63            | Bits 1–63  | Yes         |

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

Multi-character strings are OR-ed into a single KNode. Bit 0 is clear.

```
encode("ABC") → [CHAR_BIT['A'] | CHAR_BIT['B'] | CHAR_BIT['C']]
```

Properties:
- Exactly one KNode per string.
- Order is lost: `"AB"` and `"BA"` produce the same node.
- Multiplicity is lost: `"AA"` and `"A"` produce the same node.
- Suitable for signature construction and bitwise AND matching.

### Literal Encoding

Each character becomes a separate KNode. Bit 0 is set. Upper bits store
the Unicode code point.

```
encode("ABC", literal=True) → [(ord('A') << 1) | 1,
                                (ord('B') << 1) | 1,
                                (ord('C') << 1) | 1]
```

Properties:
- One KNode per character.
- Order is preserved.
- Identity is preserved (distinct characters → distinct tokens).
- Bypasses the vocabulary: any character with a valid code point can be
  encoded, including characters not in the vocabulary.
- A raw integer may also be encoded: `encode(42, literal=True)`
  → `[(42 << 1) | 1]`.

### Decoding

Auto-detects encoding mode from bit 0 of each KNode:

```
if bit 0 == 0  →  packed: find all set bits, map each to character
if bit 0 == 1  →  literal: chr(node >> 1)
```

Packed decode returns characters in bit-position order (lowest bit first),
not in the original text order.

### Signature Construction

`make_signature` performs OR-reduction over all non-literal nodes:

```
sig = 0
for node in nodes:
    if not is_literal(node):
        sig |= node
return sig
```

This produces a KSig where each set bit represents a character present in
at least one packed node. The signature supports bitwise AND matching:
a kline with signature S *signifies* a query Q if `(S & Q) != 0`.

### Literal Test

```
is_literal(node) → (node & 1) == 1
```

Bit 0 is the discriminator.

### Worked Examples

#### Packed encoding and signature (Mod32)

```
Vocabulary: "ABC" → bit 1, bit 2, bit 3

encode("A")       → [0b10]                  = [2]
encode("B")       → [0b100]                 = [4]
encode("AB")      → [0b10 | 0b100]          = [6]
encode("ABC")     → [0b10 | 0b100 | 0b1000] = [14]

make_signature([2, 4])    → 2 | 4 = 6
make_signature([2, 4, 8]) → 2 | 4 | 8 = 14
make_signature([2])       → 2
make_signature([])        → 0
```

#### Literal encoding (Mod32)

```
encode("AB", literal=True) → [(65 << 1) | 1, (66 << 1) | 1]
                            → [131, 133]

decode([131, 133])         → "AB"

is_literal(131) → True
is_literal(2)   → False
```

#### Signature excludes literals

```
nodes = [2, 131, 4]      # packed 'A', literal 'A', packed 'B'

make_signature(nodes) → 2 | 4 = 6    (131 excluded: is_literal)
```

#### Bitwise matching

```
kline_signature = make_signature([2, 4]) = 6    # "AB"
query           = make_signature([2])     = 2    # "A"

kline.signifies(query) → (6 & 2) != 0 → True   # "AB" signifies "A"
```

## What a Tokenizer is Not

The following are explicitly **out of scope** for this spec:

- **Significance computation.** Significance is defined in the
  @significance spec. The tokenizer does not compute or store significance.
- **Type prefix assignment.** How linguistic types are assigned to BPE
  tokens is an agent concern. The tokenizer provides raw token IDs.
- **Model operations.** Storing, retrieving, and querying klines are model
  concerns.
- **Training data format.** How training data is sourced and preprocessed
  for BPE training is an implementation concern.

## Referenced By

- **Kline** (@kline spec) — signatures are produced by the tokenizer.
- **Agent** (@agent spec) — uses the tokenizer to encode input, apply type
  prefixes, and construct klines.
- **Significance** (@significance spec) — consumes KNodes produced by the
  tokenizer (nodes are opaque to significance).
